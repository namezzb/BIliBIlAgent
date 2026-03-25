from http.cookies import SimpleCookie
from math import ceil
from typing import Any
from urllib.parse import urlparse

import httpx


API_ROOT = "https://api.bilibili.com"
PASSPORT_ROOT = "https://passport.bilibili.com"
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 BIliBIlAgent",
    "Accept": "application/json,text/plain,*/*",
    "Referer": "https://www.bilibili.com/",
}
VIDEO_ITEM_TYPE = 2
MAX_FAVORITE_ITEM_PAGE_SIZE = 20


class BilibiliFavoriteFolderError(RuntimeError):
    pass


class BilibiliFavoriteFolderAuthError(BilibiliFavoriteFolderError):
    pass


class BilibiliFavoriteFolderUpstreamError(BilibiliFavoriteFolderError):
    pass


class BilibiliFavoriteFolderResponseError(BilibiliFavoriteFolderError):
    pass


class BilibiliFavoriteFolderService:
    def __init__(
        self,
        *,
        api_root: str = API_ROOT,
        passport_root: str = PASSPORT_ROOT,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.api_root = api_root.rstrip("/")
        self.passport_root = passport_root.rstrip("/")
        self.timeout = timeout
        self.transport = transport

    def start_qr_login(self) -> dict[str, Any]:
        payload = self._request_json(
            "GET",
            f"{self.passport_root}/x/passport-login/web/qrcode/generate",
        )
        data = payload.get("data") or {}
        qr_url = data.get("url")
        qrcode_key = data.get("qrcode_key")
        if not qr_url or not qrcode_key:
            raise BilibiliFavoriteFolderResponseError(
                "Bilibili QR login start response is missing url or qrcode_key."
            )
        return {
            "qr_url": str(qr_url),
            "qrcode_key": str(qrcode_key),
            "expires_in_seconds": 180,
        }

    def poll_qr_login(self, qrcode_key: str) -> dict[str, Any]:
        response = self._request(
            "GET",
            f"{self.passport_root}/x/passport-login/web/qrcode/poll",
            params={"qrcode_key": qrcode_key},
        )
        payload = self._decode_payload(response)
        data = payload.get("data") or {}
        login_code = data.get("code")

        if login_code == 86101:
            return {
                "status": "pending_scan",
                "message": str(data.get("message") or "二维码未扫码。"),
                "cookie": None,
                "refresh_token": None,
                "account": None,
            }
        if login_code == 86090:
            return {
                "status": "scanned_waiting_confirm",
                "message": str(data.get("message") or "二维码已扫码，等待确认。"),
                "cookie": None,
                "refresh_token": None,
                "account": None,
            }
        if login_code == 86038:
            return {
                "status": "expired",
                "message": str(data.get("message") or "二维码已失效。"),
                "cookie": None,
                "refresh_token": None,
                "account": None,
            }
        if login_code != 0:
            raise BilibiliFavoriteFolderResponseError(
                f"Bilibili QR login poll returned unexpected status code: {login_code}"
            )

        cookie = self._extract_cookie_string(response)
        if not cookie:
            raise BilibiliFavoriteFolderResponseError(
                "Bilibili login succeeded but no Set-Cookie header was returned."
            )

        return {
            "status": "success",
            "message": str(data.get("message") or "登录成功。"),
            "cookie": cookie,
            "refresh_token": data.get("refresh_token"),
            "account": self.get_account(cookie),
        }

    def get_account(self, cookie: str) -> dict[str, Any]:
        payload = self._request_json(
            "GET",
            f"{self.api_root}/x/web-interface/nav",
            cookie=cookie,
        )
        data = payload.get("data") or {}
        if not data.get("isLogin"):
            raise BilibiliFavoriteFolderAuthError(
                "Bilibili login state is invalid or expired. Please login again."
            )

        mid = data.get("mid")
        if not mid:
            raise BilibiliFavoriteFolderResponseError(
                "Bilibili nav response is missing account mid."
            )

        return {
            "mid": int(mid),
            "uname": data.get("uname"),
            "is_login": bool(data.get("isLogin")),
        }

    def list_favorite_folders(self, cookie: str, *, folder_type: int = 2) -> dict[str, Any]:
        account = self.get_account(cookie)
        payload = self._request_json(
            "GET",
            f"{self.api_root}/x/v3/fav/folder/created/list-all",
            params={"up_mid": account["mid"], "type": folder_type},
            cookie=cookie,
        )
        folders = (payload.get("data") or {}).get("list") or []
        normalized_folders = [
            self._fetch_folder_detail(cookie, folder.get("id"), folder)
            for folder in folders
        ]
        normalized_folders.sort(key=lambda item: (item["title"], item["favorite_folder_id"]))
        return {
            "account": account,
            "total": len(normalized_folders),
            "folders": normalized_folders,
        }

    def list_folder_items(
        self,
        cookie: str,
        favorite_folder_id: str,
        *,
        pn: int = 1,
        ps: int = MAX_FAVORITE_ITEM_PAGE_SIZE,
        keyword: str = "",
        order: str = "mtime",
    ) -> dict[str, Any]:
        if pn < 1:
            raise BilibiliFavoriteFolderResponseError("pn must be at least 1.")
        if ps < 1 or ps > MAX_FAVORITE_ITEM_PAGE_SIZE:
            raise BilibiliFavoriteFolderResponseError(
                f"ps must be between 1 and {MAX_FAVORITE_ITEM_PAGE_SIZE}."
            )

        account = self.get_account(cookie)
        folder = self._fetch_folder_detail(cookie, favorite_folder_id, {"id": favorite_folder_id})
        payload = self._request_json(
            "GET",
            f"{self.api_root}/x/v3/fav/resource/list",
            params={
                "media_id": favorite_folder_id,
                "pn": pn,
                "ps": ps,
                "platform": "web",
                "order": order,
                **({"keyword": keyword} if keyword else {}),
            },
            cookie=cookie,
        )
        data = payload.get("data") or {}
        info = data.get("info") or {}
        total = int(info.get("media_count") or folder.get("media_count") or 0)
        total_pages = ceil(total / ps) if total else 1
        items = [
            self._normalize_folder_item(item, favorite_folder_id=str(folder["favorite_folder_id"]))
            for item in (data.get("medias") or [])
        ]
        return {
            "account": account,
            "folder": folder,
            "page": pn,
            "page_size": ps,
            "total": total,
            "total_pages": total_pages,
            "has_more": pn < total_pages,
            "items": items,
        }

    def list_all_folder_items(
        self,
        cookie: str,
        favorite_folder_id: str,
        *,
        keyword: str = "",
        order: str = "mtime",
    ) -> dict[str, Any]:
        first_page = self.list_folder_items(
            cookie,
            favorite_folder_id,
            pn=1,
            ps=MAX_FAVORITE_ITEM_PAGE_SIZE,
            keyword=keyword,
            order=order,
        )
        total_pages = int(first_page["total_pages"])
        items = list(first_page["items"])
        for page in range(2, total_pages + 1):
            payload = self.list_folder_items(
                cookie,
                favorite_folder_id,
                pn=page,
                ps=MAX_FAVORITE_ITEM_PAGE_SIZE,
                keyword=keyword,
                order=order,
            )
            items.extend(payload["items"])

        first_page["page"] = 1
        first_page["page_size"] = MAX_FAVORITE_ITEM_PAGE_SIZE
        first_page["has_more"] = False
        first_page["items"] = items
        return first_page

    def get_video_view(
        self,
        cookie: str,
        *,
        bvid: str | None = None,
        aid: int | None = None,
    ) -> dict[str, Any]:
        if not bvid and not aid:
            raise BilibiliFavoriteFolderResponseError("Either bvid or aid is required.")
        payload = self._request_json(
            "GET",
            f"{self.api_root}/x/web-interface/view",
            params={key: value for key, value in {"bvid": bvid, "aid": aid}.items() if value is not None},
            cookie=cookie,
        )
        data = payload.get("data")
        if not isinstance(data, dict):
            raise BilibiliFavoriteFolderResponseError("Bilibili view response is missing data.")
        return data

    def get_playurl(
        self,
        cookie: str,
        *,
        bvid: str,
        cid: int,
    ) -> dict[str, Any]:
        payload = self._request_json(
            "GET",
            f"{self.api_root}/x/player/playurl",
            params={
                "bvid": bvid,
                "cid": cid,
                "fnval": 4048,
                "qn": 80,
                "fourk": 1,
            },
            cookie=cookie,
        )
        data = payload.get("data")
        if not isinstance(data, dict):
            raise BilibiliFavoriteFolderResponseError("Bilibili playurl response is missing data.")
        return data

    def fetch_subtitle_body(self, subtitle_url: str, *, cookie: str | None = None) -> dict[str, Any]:
        normalized_url = self._normalize_subtitle_url(subtitle_url)
        response = self._request("GET", normalized_url, cookie=cookie)
        payload = self._decode_payload(response)
        body = payload.get("body")
        if body is None:
            raise BilibiliFavoriteFolderResponseError(
                "Bilibili subtitle response is missing body."
            )
        return payload

    def choose_subtitle_entry(self, subtitles: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not subtitles:
            return None

        def score(entry: dict[str, Any]) -> tuple[int, int]:
            language = str(entry.get("lan") or "").lower()
            if language in {"zh-cn", "zh-hans"}:
                return (0, 0)
            if language.startswith("zh"):
                return (1, 0)
            return (2, 0)

        return sorted(subtitles, key=score)[0]

    def subtitle_payload_to_blocks(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        body = payload.get("body")
        if not isinstance(body, list):
            raise BilibiliFavoriteFolderResponseError("Bilibili subtitle body is invalid.")
        blocks: list[dict[str, Any]] = []
        for index, item in enumerate(body):
            if not isinstance(item, dict):
                continue
            text = str(item.get("content") or "").strip()
            if not text:
                continue
            blocks.append(
                {
                    "text": text,
                    "start_ms": self._seconds_to_ms(item.get("from")),
                    "end_ms": self._seconds_to_ms(item.get("to")),
                    "block_index": index,
                }
            )
        return blocks

    def _fetch_folder_detail(
        self,
        cookie: str,
        folder_id: Any,
        fallback: dict[str, Any],
    ) -> dict[str, Any]:
        if not folder_id:
            raise BilibiliFavoriteFolderResponseError(
                "Received a favorite folder without an id."
            )
        payload = self._request_json(
            "GET",
            f"{self.api_root}/x/v3/fav/folder/info",
            params={"media_id": folder_id},
            cookie=cookie,
        )
        detail = payload.get("data") or {}
        owner = detail.get("upper") or {}
        return {
            "favorite_folder_id": str(detail.get("id") or folder_id),
            "title": str(detail.get("title") or fallback.get("title") or ""),
            "intro": detail.get("intro") or fallback.get("intro"),
            "cover": detail.get("cover") or fallback.get("cover"),
            "media_count": int(detail.get("media_count") or fallback.get("media_count") or 0),
            "folder_attr": detail.get("attr", fallback.get("attr")),
            "owner_mid": owner.get("mid") or detail.get("mid") or fallback.get("mid"),
        }

    def _normalize_folder_item(
        self,
        item: dict[str, Any],
        *,
        favorite_folder_id: str,
    ) -> dict[str, Any]:
        bvid = item.get("bvid") or item.get("bv_id")
        aid = item.get("id") or item.get("aid")
        item_type = int(item.get("type") or 0)
        video_id = self._preferred_video_id(item)
        selectable = item_type == VIDEO_ITEM_TYPE and bool(video_id)
        unsupported_reason = None
        if not selectable:
            if item_type != VIDEO_ITEM_TYPE:
                unsupported_reason = f"Unsupported favorite item type: {item_type}"
            else:
                unsupported_reason = "Missing video identifier in favorite item."
        upper = item.get("upper") or {}
        return {
            "item_id": str(item.get("id") or video_id or ""),
            "favorite_folder_id": favorite_folder_id,
            "item_type": item_type,
            "media_type": item_type,
            "selectable": selectable,
            "unsupported_reason": unsupported_reason,
            "video_id": video_id,
            "aid": int(aid) if aid is not None else None,
            "bvid": str(bvid) if bvid else None,
            "title": str(item.get("title") or ""),
            "cover": item.get("cover"),
            "intro": item.get("intro") or item.get("desc"),
            "duration": int(item.get("duration") or 0),
            "upper_mid": upper.get("mid"),
            "upper_name": upper.get("name"),
            "fav_time": item.get("fav_time"),
            "pubtime": item.get("pubtime"),
            "raw": item,
        }

    def _preferred_video_id(self, item: dict[str, Any]) -> str | None:
        bvid = item.get("bvid") or item.get("bv_id")
        if bvid:
            return str(bvid)
        aid = item.get("id") or item.get("aid")
        if aid is not None:
            return str(aid)
        return None

    def _normalize_subtitle_url(self, subtitle_url: str) -> str:
        parsed = urlparse(subtitle_url)
        if parsed.scheme:
            return subtitle_url
        if subtitle_url.startswith("//"):
            return f"https:{subtitle_url}"
        if subtitle_url.startswith("/"):
            return f"https://{parsed.netloc}{subtitle_url}" if parsed.netloc else f"https://api.bilibili.com{subtitle_url}"
        return f"https://{subtitle_url}"

    def _seconds_to_ms(self, value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(float(value) * 1000)
        except (TypeError, ValueError):
            return None

    def _request_json(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        cookie: str | None = None,
    ) -> dict[str, Any]:
        response = self._request(method, url, params=params, cookie=cookie)
        payload = self._decode_payload(response)
        code = payload.get("code")
        if code not in (None, 0):
            message = str(payload.get("message") or payload.get("msg") or "Unknown Bilibili API error.")
            if code == -101:
                raise BilibiliFavoriteFolderAuthError(
                    "Bilibili login state is invalid or expired. Please login again."
                )
            raise BilibiliFavoriteFolderUpstreamError(
                f"Bilibili API error code={code}: {message}"
            )
        return payload

    def _request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        cookie: str | None = None,
    ) -> httpx.Response:
        headers = dict(DEFAULT_HEADERS)
        if cookie:
            headers["Cookie"] = cookie
        try:
            with httpx.Client(
                headers=headers,
                timeout=self.timeout,
                follow_redirects=True,
                transport=self.transport,
            ) as client:
                response = client.request(method, url, params=params)
        except httpx.TimeoutException as exc:
            raise BilibiliFavoriteFolderUpstreamError(
                "Bilibili request timed out."
            ) from exc
        except httpx.HTTPError as exc:
            raise BilibiliFavoriteFolderUpstreamError(
                f"Bilibili request failed: {exc}"
            ) from exc

        if response.status_code >= 500:
            raise BilibiliFavoriteFolderUpstreamError(
                f"Bilibili upstream service returned {response.status_code}."
            )
        if response.status_code >= 400:
            raise BilibiliFavoriteFolderResponseError(
                f"Bilibili request was rejected with status {response.status_code}."
            )
        return response

    def _decode_payload(self, response: httpx.Response) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise BilibiliFavoriteFolderResponseError(
                "Bilibili returned invalid JSON."
            ) from exc
        if not isinstance(payload, dict):
            raise BilibiliFavoriteFolderResponseError(
                "Bilibili returned an unexpected payload shape."
            )
        return payload

    def _extract_cookie_string(self, response: httpx.Response) -> str:
        cookie_names: dict[str, str] = {}
        for header in response.headers.get_list("set-cookie"):
            parsed = SimpleCookie()
            parsed.load(header)
            for morsel in parsed.values():
                cookie_names[morsel.key] = morsel.value
        if not cookie_names:
            return ""
        return "; ".join(f"{key}={value}" for key, value in sorted(cookie_names.items()))
