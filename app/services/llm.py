import httpx
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


def _embed_via_dashscope(api_key: str, model: str, texts: list[str]) -> list[list[float]]:
    """Call DashScope TextEmbedding API for document indexing."""
    import dashscope
    from dashscope import TextEmbedding
    dashscope.api_key = api_key
    response = TextEmbedding.call(
        model=model,
        input=texts,
        text_type="document",
    )
    if response.status_code != 200:
        raise RuntimeError(f"DashScope embedding failed: {response.code} {response.message}")
    items = sorted(response.output["embeddings"], key=lambda e: e["text_index"])
    return [list(item["embedding"]) for item in items]


class OpenAICompatibleLLM:
    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str | None,
        model: str,
        summary_model: str | None,
        embedding_model: str,
        system_prompt: str,
        embedding_api_key: str | None = None,
        embedding_base_url: str | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.summary_model = summary_model or model
        self.embedding_model = embedding_model
        self.system_prompt = system_prompt
        self.embedding_api_key = embedding_api_key or api_key
        self.embedding_base_url = embedding_base_url
        self._client: OpenAI | None = None
        self._embed_client: OpenAI | None = None
        self._lc_chat: ChatOpenAI | None = None

    def _get_client(self) -> OpenAI:
        if self._client is None:
            client_kwargs: dict = {
                "http_client": httpx.Client(proxy=None, trust_env=False),
            }
            if self.api_key:
                client_kwargs["api_key"] = self.api_key
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            self._client = OpenAI(**client_kwargs)
        return self._client

    # Aliases for backward compatibility with agent/service.py
    def get_langchain_llm(self) -> ChatOpenAI:
        return self.get_lc_chat_model()

    def _build_lc_messages(self, messages: list[dict[str, str]], *, extra_system_messages: list[str] | None = None) -> list:
        lc: list = [SystemMessage(content=self.system_prompt)]
        for content in (extra_system_messages or []):
            if content.strip():
                lc.append(SystemMessage(content=content))
        for m in messages:
            role = m.get("role", "user")
            if role == "user":
                lc.append(HumanMessage(content=m.get("content", "")))
            elif role == "assistant":
                lc.append(AIMessage(content=m.get("content", "")))
            else:
                lc.append(SystemMessage(content=m.get("content", "")))
        return lc

    def get_lc_chat_model(self) -> ChatOpenAI:
        """Return a LangChain ChatOpenAI instance for use inside LangGraph nodes.

        LangGraph astream(stream_mode='messages') will automatically intercept
        this model's output and stream tokens — no need to call .stream() manually.
        """
        if self._lc_chat is None:
            kwargs: dict = {
                "model": self.model,
                "temperature": 0.2,
                "timeout": 10,
                "http_client": httpx.Client(proxy=None, trust_env=False),
                "http_async_client": httpx.AsyncClient(proxy=None, trust_env=False),
            }
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._lc_chat = ChatOpenAI(**kwargs)
        return self._lc_chat

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        extra_system_messages: list[str] | None = None,
    ) -> str:
        if not self.api_key:
            return self._fallback_chat(messages)

        system_messages = [{"role": "system", "content": self.system_prompt}]
        if extra_system_messages:
            system_messages.extend(
                {"role": "system", "content": content}
                for content in extra_system_messages
                if content.strip()
            )

        try:
            content = self._create_chat_completion(
                messages=[*system_messages, *messages],
                model=self.model,
                temperature=0.2,
            )
        except Exception as exc:
            return f"LLM request failed: {exc}"

        if isinstance(content, str) and content.strip():
            return content

        return "The model returned an empty response."

    def chat_lc(
        self,
        messages: list[dict[str, str]],
        *,
        extra_system_messages: list[str] | None = None,
    ) -> str:
        """Chat using the LangChain ChatOpenAI model (supports streaming via LangGraph)."""
        if not self.api_key:
            return self._fallback_chat(messages)
        lc_messages = self._build_lc_messages(
            messages, extra_system_messages=extra_system_messages
        )
        try:
            result = self.get_lc_chat_model().invoke(lc_messages)
            return result.content or "The model returned an empty response."
        except Exception as exc:
            return f"LLM request failed: {exc}"

    def summarize_conversation(self, messages: list[dict[str, str]]) -> str | None:
        if not self.api_key:
            return None
        if not messages:
            return None

        transcript = "\n".join(
            f"{message['role'].upper()}: {message['content']}" for message in messages
        )
        summary_prompt = (
            "Summarize the prior conversation for future turns. "
            "Keep only stable facts, unresolved goals, and important constraints. "
            "Do not include roleplay, filler, or formatting. "
            "Return plain text in at most 6 short lines."
        )

        try:
            content = self._create_chat_completion(
                messages=[
                    {"role": "system", "content": summary_prompt},
                    {
                        "role": "user",
                        "content": f"Conversation transcript:\n{transcript}",
                    },
                ],
                model=self.summary_model,
                temperature=0.1,
            )
        except Exception:
            return None

        if isinstance(content, str) and content.strip():
            return content.strip()

        return None

    def _get_embed_client(self) -> OpenAI:
        if self._embed_client is None:
            client_kwargs: dict = {
                "http_client": httpx.Client(proxy=None, trust_env=False),
            }
            # Use dedicated embedding key/url if provided, else fall back to main LLM config
            api_key = self.embedding_api_key or self.api_key
            base_url = self.embedding_base_url or self.base_url
            if api_key:
                client_kwargs["api_key"] = api_key
            if base_url:
                client_kwargs["base_url"] = base_url
            self._embed_client = OpenAI(**client_kwargs)
        return self._embed_client

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        api_key = self.embedding_api_key or self.api_key
        if not api_key:
            raise RuntimeError("EMBEDDING_API_KEY or OPENROUTER_API_KEY is required for embeddings.")

        # DashScope models (e.g. text-embedding-v3) don't use OpenAI-compatible endpoint
        _DASHSCOPE_MODELS = {"text-embedding-v1", "text-embedding-v2", "text-embedding-v3"}
        if self.embedding_model in _DASHSCOPE_MODELS:
            return _embed_via_dashscope(api_key, self.embedding_model, texts)

        response = self._get_embed_client().embeddings.create(
            model=self.embedding_model,
            input=texts,
            encoding_format="float",
            extra_headers={
                "HTTP-Referer": "https://github.com/BIliBIlAgent",
                "X-Title": "BIliBIlAgent",
            },
        )
        return [list(item.embedding) for item in response.data]

    def _create_chat_completion(
        self,
        *,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
    ) -> str | None:
        completion = self._get_client().chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            timeout=10.0,
        )
        return completion.choices[0].message.content

    def _fallback_chat(self, messages: list[dict[str, str]]) -> str:
        user_messages = [message["content"] for message in messages if message["role"] == "user"]
        latest_user_message = user_messages[-1].lower() if user_messages else ""

        if len(user_messages) >= 2 and latest_user_message in {
            "我刚刚问了什么？",
            "我刚刚问了什么?",
            "我刚才问了什么？",
            "我刚才问了什么?",
            "what did i just ask?",
            "what was my previous question?",
        }:
            return f"Your previous question was: {user_messages[-2]}"

        return (
            "General chat is available, but no LLM provider is configured yet. "
            "Set LLM_API_KEY to enable model responses."
        )


