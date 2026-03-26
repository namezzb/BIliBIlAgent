# BIliBIlAgent 前端对接文档

> 版本：2026-03-26  
> 后端基础地址：`http://localhost:8000`（开发环境）  
> 所有接口均以 `/api` 为前缀，除 `/` 和 `/health` 外。

---

## 目录

1. [通用约定](#1-通用约定)
2. [系统接口](#2-系统接口)
3. [Agent 对话接口](#3-agent-对话接口)
4. [Run 运行管理接口](#4-run-运行管理接口)
5. [会话管理接口](#5-会话管理接口)
6. [用户长期记忆接口](#6-用户长期记忆接口)
7. [B 站登录接口](#7-b-站登录接口)
8. [B 站收藏夹接口](#8-b-站收藏夹接口)
9. [知识库接口](#9-知识库接口)
10. [前端页面功能描述](#10-前端页面功能描述)
11. [核心数据流示意](#11-核心数据流示意)

---

## 1. 通用约定

### 请求格式
- Content-Type: `application/json`
- B 站登录态通过请求头传递：`X-Bilibili-Cookie: <cookie_string>`

### 响应格式
所有接口返回 JSON，错误时返回：
```json
{ "detail": "错误描述" }
```

### 常见 HTTP 状态码
| 状态码 | 含义 |
|--------|------|
| 200 | 成功 |
| 202 | 已接受（异步任务已提交）|
| 400 | 请求参数错误 |
| 401 | 未登录 / Cookie 失效 |
| 404 | 资源不存在 |
| 409 | 冲突（如重复导入）|
| 500 | 服务端错误 |
| 502 | B 站上游响应异常 |
| 503 | 上游服务不可用 |

### 关键字段说明
| 字段 | 说明 |
|------|------|
| `session_id` | 会话 ID，同一对话窗口复用同一值，首次可不传由后端生成 |
| `user_id` | 用户 ID，用于跨会话持久化长期记忆，可选 |
| `run_id` | 单次 Agent 运行 ID，每轮对话或每次导入各生成一个 |
| `route` | 路由类型：`general_chat` / `favorite_knowledge_query` / `video_knowledge_query` / `import_request` / `retry_request` |
| `status` | 运行状态：`running` / `awaiting_confirmation` / `completed` / `failed` / `cancelled` |

---

## 2. 系统接口

### GET `/`
服务根节点，可用于前端检查后端是否在线。

**响应**
```json
{
  "name": "BIliBIlAgent",
  "status": "ready",
  "message": "Use /health to verify service status."
}
```

---

### GET `/health`
健康检查。

**响应**
```json
{
  "status": "ok",
  "service": "bilibilagent-backend"
}
```

---

## 3. Agent 对话接口

### POST `/api/chat`
发送一条用户消息，返回 Agent 的同步回复。

**请求体**
```json
{
  "session_id": "string | null",
  "user_id": "string | null",
  "message": "string (必填，最短1字符)"
}
```

**响应 `ChatResponse`**
```json
{
  "session_id": "string",
  "run_id": "string",
  "intent": "string | null",
  "route": "general_chat | favorite_knowledge_query | video_knowledge_query | import_request | retry_request | null",
  "langsmith_thread_id": "string | null",
  "langsmith_thread_url": "string | null",
  "status": "running | awaiting_confirmation | completed | failed",
  "reply": "string",
  "requires_confirmation": false,
  "approval_status": "string | null",
  "execution_plan": "ExecutionPlan | null",
  "pending_actions": []
}
```

**ExecutionPlan 结构**（当 `requires_confirmation=true` 时出现）
```json
{
  "goal": "string",
  "summary": "string",
  "steps": [
    {
      "id": "string",
      "title": "string",
      "description": "string",
      "tool": "string | null",
      "action": "string | null",
      "status": "string"
    }
  ],
  "tool_calls": [
    {
      "tool": "string",
      "action": "string",
      "description": "string",
      "target": "string",
      "args": {},
      "side_effect": true
    }
  ]
}
```

**关键逻辑**
- 若 `status=awaiting_confirmation`，前端需展示 `execution_plan` 并提供确认/取消按钮，然后调用 `POST /api/runs/{run_id}/confirm`。
- 若 `status=completed`，直接展示 `reply`。
- `session_id` 首次不传时由后端生成，前端需从响应中保存并在后续请求中携带。

---

### POST `/api/runs/{run_id}/confirm`
对需要确认的 Agent 执行计划给出批准或拒绝。

**路径参数**
- `run_id`：要确认的运行 ID

**请求体**
```json
{ "approved": true }
```

**响应**：同 `ChatResponse`（status 变为 `completed` 或 `failed`）

**错误**
- `404`：run 不存在
- `409`：run 当前状态不是 `awaiting_confirmation`

---

## 4. Run 运行管理接口

### GET `/api/runs/{run_id}`
获取某次运行的完整详情，包含执行步骤。

**响应 `RunDetailResponse`**（继承自 `ChatResponse`，额外字段：）
```json
{
  "...": "继承 ChatResponse 全部字段",
  "approval_requested_at": "ISO8601 | null",
  "approval_resolved_at": "ISO8601 | null",
  "created_at": "ISO8601",
  "updated_at": "ISO8601",
  "event_count": 0,
  "steps": [
    {
      "step_key": "string",
      "step_name": "string",
      "status": "completed | failed | running",
      "input_summary": "string | null",
      "output_summary": "string | null",
      "updated_at": "ISO8601"
    }
  ]
}
```

---

### GET `/api/runs/{run_id}/events`
SSE 实时事件流，订阅某次运行的执行事件。

**查询参数**
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `follow` | bool | `true` | `true` 时持续推送直到运行结束；`false` 时仅回放已有事件后关闭 |

**SSE 事件格式**
```
id: <sequence>
event: run_event
data: {"event_id":"...","run_id":"...","sequence":1,"type":"...","timestamp":"...","payload":{...}}
```

**事件类型（`type` 字段）**
| type | 含义 |
|------|------|
| `run_started` | 运行开始 |
| `context_loaded` | 会话上下文加载完成 |
| `intent_classified` | 意图/路由分类完成 |
| `response_prepared` | 回复准备完成 |
| `confirmation_requested` | 需要用户确认 |
| `tool_executing` | 工具执行中 |
| `run_completed` | 运行成功完成 |
| `run_failed` | 运行失败 |

**响应头**
```
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive
```

---

## 5. 会话管理接口

### GET `/api/sessions/{session_id}`
获取会话详情，包含消息历史、摘要和最近上下文。

**响应 `SessionDetailResponse`**
```json
{
  "session_id": "string",
  "user_id": "string | null",
  "summary_text": "string | null",
  "recent_context": {
    "last_retrieval": {},
    "last_run_route": "string"
  },
  "created_at": "ISO8601",
  "updated_at": "ISO8601",
  "messages": [
    {
      "message_id": "string",
      "run_id": "string | null",
      "role": "user | assistant",
      "content": "string",
      "created_at": "ISO8601"
    }
  ]
}
```

**错误**
- `404`：session 不存在

---

## 6. 用户长期记忆接口

### GET `/api/users/{user_id}/memory`
获取用户的长期记忆档案。

**响应 `UserMemoryProfileResponse`**
```json
{
  "user_id": "string",
  "preferences": {
    "回答语言": {
      "value": "中文",
      "source_type": "chat | api",
      "source_run_id": "string | null",
      "source_text": "string | null",
      "confirmed": true,
      "created_at": "ISO8601",
      "updated_at": "ISO8601"
    }
  },
  "aliases": {},
  "default_scopes": {},
  "created_at": "ISO8601 | null",
  "updated_at": "ISO8601 | null"
}
```

**三个记忆分组说明**
| 分组 | 用途 |
|------|------|
| `preferences` | 用户偏好（语言、风格等）|
| `aliases` | 用户自定义别名（如"我的技术收藏夹"）|
| `default_scopes` | 默认检索范围（如默认收藏夹 ID）|

---

### PATCH `/api/users/{user_id}/memory`
批量更新/新增用户记忆条目。

**请求体**
```json
{
  "preferences": { "key": "value" },
  "aliases": { "key": "value" },
  "default_scopes": { "key": "value" }
}
```
三个字段均可选，至少传一个。value 为字符串。

**响应**：同 `GET` 的 `UserMemoryProfileResponse`

---

### DELETE `/api/users/{user_id}/memory/{group}/{key}`
删除某条记忆。

**路径参数**
- `group`：`preferences` / `aliases` / `default_scopes`
- `key`：记忆条目的键名

**响应**：同 `GET` 的 `UserMemoryProfileResponse`

---

## 7. B 站登录接口

### POST `/api/bilibili/auth/qr/start`
生成 B 站扫码登录二维码。

**请求体**：无

**响应**
```json
{
  "qr_url": "https://www.bilibili.com/...",
  "qrcode_key": "string",
  "expires_in_seconds": 180
}
```

**前端操作**：将 `qr_url` 渲染为二维码图片（推荐使用 qrcode 库生成），然后轮询下方接口。

---

### GET `/api/bilibili/auth/qr/poll?qrcode_key={key}`
轮询扫码状态。

**查询参数**
- `qrcode_key`（必填）：从 `start` 接口获取的 key

**响应**
```json
{
  "status": "pending_scan | scanned_waiting_confirm | expired | success",
  "message": "string",
  "cookie": "string | null",
  "refresh_token": "string | null",
  "account": {
    "mid": 12345678,
    "uname": "用户名",
    "is_login": true
  }
}
```

**状态说明**
| status | 含义 | 前端处理 |
|--------|------|----------|
| `pending_scan` | 等待扫码 | 继续轮询 |
| `scanned_waiting_confirm` | 已扫码，等待确认 | 提示用户在手机上确认 |
| `expired` | 二维码已过期 | 提示刷新，重新调用 start |
| `success` | 登录成功 | 保存 `cookie`，停止轮询 |

**重要**：`status=success` 时的 `cookie` 字段需由前端持久化（localStorage 或安全 Cookie），后续所有 B 站相关接口均需在请求头中携带：
```
X-Bilibili-Cookie: <cookie 字符串>
```

---

## 8. B 站收藏夹接口

> 以下接口均需请求头 `X-Bilibili-Cookie`，否则返回 `401`。

### GET `/api/bilibili/favorite-folders`
获取当前登录用户的全部收藏夹列表。

**请求头**：`X-Bilibili-Cookie: <cookie>`

**响应**
```json
{
  "account": {
    "mid": 12345678,
    "uname": "用户名",
    "is_login": true
  },
  "total": 3,
  "folders": [
    {
      "favorite_folder_id": "123456789",
      "title": "我的收藏夹",
      "intro": "简介",
      "cover": "https://...",
      "media_count": 42,
      "folder_attr": 0,
      "owner_mid": 12345678
    }
  ]
}
```

**错误**
- `401`：Cookie 缺失或失效，需重新登录
- `503`：B 站服务不可用

---

### GET `/api/bilibili/favorite-folders/{favorite_folder_id}/videos`
获取某个收藏夹内的视频列表（分页）。

**路径参数**
- `favorite_folder_id`：收藏夹 ID

**请求头**：`X-Bilibili-Cookie: <cookie>`

**查询参数**
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `pn` | int | 1 | 页码（≥1）|
| `ps` | int | 20 | 每页数量（1-20）|
| `keyword` | string | "" | 搜索关键词 |
| `order` | string | `mtime` | 排序方式：`mtime`（收藏时间）/ `view`（播放量）/ `pubtime`（发布时间）|

**响应**
```json
{
  "account": { "mid": 12345678, "uname": "用户名", "is_login": true },
  "folder": {
    "favorite_folder_id": "123456789",
    "title": "我的收藏夹",
    "intro": "简介",
    "cover": "https://...",
    "media_count": 42,
    "folder_attr": 0,
    "owner_mid": 12345678
  },
  "page": 1,
  "page_size": 20,
  "total": 42,
  "total_pages": 3,
  "has_more": true,
  "items": [
    {
      "item_id": "string",
      "favorite_folder_id": "123456789",
      "item_type": 2,
      "media_type": 2,
      "selectable": true,
      "unsupported_reason": null,
      "video_id": "string",
      "aid": 123456,
      "bvid": "BV1xx411c7mD",
      "title": "视频标题",
      "cover": "https://...",
      "intro": "视频简介",
      "duration": 360,
      "upper_mid": 987654,
      "upper_name": "UP主名",
      "fav_time": 1711382400,
      "pubtime": 1711296000
    },
    {
      "item_id": "string",
      "selectable": false,
      "unsupported_reason": "非视频类型收藏项，不支持导入",
      "title": "某专栏文章"
    }
  ]
}
```

**字段说明**
- `selectable=false` 的条目为非视频类型（专栏、音频等），前端应将其置灰并显示 `unsupported_reason`
- `duration` 单位为秒
- `fav_time` / `pubtime` 为 Unix 时间戳（秒）

---

### POST `/api/bilibili/imports`
提交导入任务，异步执行字幕获取、ASR 准备和知识库写入。

**请求头**：`X-Bilibili-Cookie: <cookie>`

**请求体**
```json
{
  "session_id": "string | null",
  "user_id": "string | null",
  "favorite_folder_id": "123456789",
  "selected_video_ids": ["video_id_1", "video_id_2"]
}
```

**响应**（HTTP 202 Accepted）：同 `ChatResponse`
```json
{
  "session_id": "string",
  "run_id": "string",
  "status": "running",
  "reply": "Import accepted. The backend is fetching subtitles...",
  "route": "import_request",
  "requires_confirmation": false,
  "execution_plan": { "goal": "...", "summary": "...", "steps": [], "tool_calls": [] }
}
```

**后续跟踪**：使用返回的 `run_id` 轮询 `GET /api/runs/{run_id}` 或订阅 `GET /api/runs/{run_id}/events` 获取导入进度。

**错误**
- `400`：`selected_video_ids` 中含有不属于该收藏夹的视频
- `401`：Cookie 缺失或失效

---

## 9. 知识库接口

### POST `/api/knowledge/search`
在已导入的知识库中搜索相关内容片段。

**请求体**
```json
{
  "query": "这个视频讲了什么",
  "top_k": 5,
  "favorite_folder_ids": [],
  "video_ids": [],
  "page_numbers": [],
  "source_types": []
}
```

**字段说明**
| 字段 | 类型 | 说明 |
|------|------|------|
| `query` | string | 搜索查询文本（必填）|
| `top_k` | int | 返回结果数，1-20，默认 5 |
| `favorite_folder_ids` | string[] | 限定收藏夹范围，空数组=全库 |
| `video_ids` | string[] | 限定视频范围 |
| `page_numbers` | int[] | 限定分 P 范围 |
| `source_types` | string[] | `subtitle` / `asr`，空=不限 |

**响应**
```json
{
  "query": "这个视频讲了什么",
  "total_hits": 3,
  "hits": [
    {
      "score": 0.92,
      "chunk_id": "string",
      "text": "...片段文本...",
      "source_type": "subtitle",
      "source_language": "zh-CN",
      "start_ms": 12000,
      "end_ms": 18000,
      "video": {
        "video_id": "string",
        "bvid": "BV1xx411c7mD",
        "title": "视频标题"
      },
      "favorite_folders": [
        { "favorite_folder_id": "123", "title": "收藏夹名", "intro": null }
      ],
      "pages": [
        { "page_id": "string", "page_number": 1, "title": "第一P" }
      ]
    }
  ]
}
```

---

### POST `/api/knowledge/debug/index`
（调试用）手动向知识库写入结构化数据，用于测试检索效果。生产环境前端无需调用此接口，导入通过 `/api/bilibili/imports` 触发。

---

## 10. 前端页面功能描述

---

### 页面一：登录页（B 站扫码登录）

**目的**：引导用户完成 B 站账号授权，获取并持久化 Cookie。

**功能点**
1. 调用 `POST /api/bilibili/auth/qr/start` 获取二维码数据，将 `qr_url` 渲染为二维码图片。
2. 二维码显示倒计时（180 秒），过期后展示「刷新」按钮重新获取。
3. 每 2 秒轮询 `GET /api/bilibili/auth/qr/poll?qrcode_key=xxx`：
   - `pending_scan`：显示「等待扫码」
   - `scanned_waiting_confirm`：显示「请在手机上确认登录」
   - `expired`：停止轮询，提示刷新二维码
   - `success`：停止轮询，将 `cookie` 存入 localStorage，跳转至收藏夹页
4. 登录成功后展示账号名（`account.uname`）和头像占位。

**状态持久化**：`biliCookie`（string）存入 localStorage，页面加载时检查是否存在，存在则直接跳过登录页。

---

### 页面二：收藏夹列表页（导入入口）

**目的**：展示用户在 B 站的所有收藏夹，供用户选择要导入的收藏夹。

**功能点**
1. 页面加载时从 localStorage 读取 Cookie，调用 `GET /api/bilibili/favorite-folders`（请求头携带 Cookie）获取收藏夹列表。
2. 以卡片形式展示每个收藏夹：封面图、名称、视频数量、简介（可折叠）。
3. 点击收藏夹卡片，进入「视频选择页」。
4. 若返回 `401`，清除本地 Cookie 并跳转至登录页。
5. 右上角显示当前登录的账号名。
6. 提供「退出登录」功能（清除 localStorage 中的 Cookie，跳转至登录页）。

---

### 页面三：视频选择页（导入视频勾选）

**目的**：展示收藏夹内的视频列表，支持勾选部分或全部视频发起导入。

**功能点**
1. 调用 `GET /api/bilibili/favorite-folders/{favorite_folder_id}/videos` 获取视频列表，支持分页（每页 20 条）。
2. 每个视频展示：封面图、标题、UP 主名、时长、收藏时间。
3. `selectable=false` 的条目（专栏、音频等）灰色置底显示，展示 `unsupported_reason`，不可勾选。
4. 支持全选、反选、取消全选操作。
5. 支持按关键词搜索过滤（传 `keyword` 参数）。
6. 支持切换排序方式：收藏时间 / 播放量 / 发布时间（对应 `order` 参数）。
7. 底部固定「导入选中视频」按钮，显示已选数量，点击后：
   - 调用 `POST /api/bilibili/imports`（请求头带 Cookie，Body 带 `favorite_folder_id` + `selected_video_ids`）
   - 获取 `run_id` 后跳转至「导入进度页」

---

### 页面四：导入进度页

**目的**：实时展示导入任务的进度和每个视频的处理结果。

**功能点**
1. 从路由参数获取 `run_id`，订阅 `GET /api/runs/{run_id}/events`（SSE）实时接收事件。
2. 根据事件 `type` 更新 UI：
   - `run_started`：显示「导入开始」
   - `tool_executing`：显示当前正在处理的视频标题
   - `run_completed`：显示「导入完成」汇总
   - `run_failed`：显示错误信息
3. 同时轮询 `GET /api/runs/{run_id}`（每 3 秒一次）获取完整步骤列表（`steps`），展示每个步骤的状态。
4. 导入完成后展示汇总：
   - 成功索引视频数
   - 跳过（重复）视频数
   - 需 ASR 处理的视频数（`needs_asr=true`）
   - 失败视频数及失败原因
5. 提供「返回收藏夹」和「去对话」快捷跳转按钮。

---

### 页面五：对话页（主聊天界面）

**目的**：与 Agent 进行多轮对话，支持知识问答和任务执行。

**功能点**

#### 5.1 基础对话
1. 页面加载时生成或恢复 `session_id`（存入 localStorage），可选绑定 `user_id`。
2. 消息输入框 + 发送按钮，调用 `POST /api/chat`。
3. 展示消息气泡：用户消息靠右，Agent 回复靠左，支持 Markdown 渲染。
4. 发送后显示加载动画，收到响应后渲染回复内容。
5. 展示 Agent 回复来源标签（`route` 字段）：
   - `general_chat`：普通对话
   - `favorite_knowledge_query`：收藏夹知识问答
   - `video_knowledge_query`：单视频知识问答
   - `import_request`：导入任务
   - `retry_request`：重试任务

#### 5.2 执行确认流
1. 当 `status=awaiting_confirmation` 时，在消息气泡下方展示执行计划卡片：
   - 显示 `execution_plan.goal`（本次目标）
   - 显示 `execution_plan.summary`（摘要）
   - 列出 `execution_plan.steps`（执行步骤列表）
   - 列出 `execution_plan.tool_calls`（将调用的工具，标注 `side_effect=true` 的为有副作用操作）
2. 提供「确认执行」和「取消」两个按钮：
   - 确认：`POST /api/runs/{run_id}/confirm` with `{"approved": true}`
   - 取消：`POST /api/runs/{run_id}/confirm` with `{"approved": false}`
3. 确认/取消后禁用按钮并展示最终回复。

#### 5.3 知识来源展示
1. 知识问答回复中，解析回复文本里的「来源：」行，以标签或卡片形式展示来源视频标题和分P信息。
2. 可选：在回复气泡旁提供「查看来源」展开按钮，调用 `POST /api/knowledge/search` 展示原始检索片段。

#### 5.4 会话管理
1. 侧边栏或顶部菜单支持「新建会话」（清除本地 `session_id`，刷新页面）。
2. 可选：调用 `GET /api/sessions/{session_id}` 展示当前会话的历史消息，支持滚动查看上下文。

---

### 页面六：长期记忆管理页（设置面板）

**目的**：让用户查看、修改和删除 Agent 为其记录的长期偏好。

**功能点**
1. 需要 `user_id`（从设置或首次使用时填写，存入 localStorage）。
2. 调用 `GET /api/users/{user_id}/memory` 展示三组记忆：
   - **偏好（preferences）**：如回答语言、回答风格
   - **别名（aliases）**：如「我的技术收藏夹」→ 真实收藏夹 ID
   - **默认范围（default_scopes）**：默认检索的收藏夹
3. 每条记忆显示：键名、值、来源类型（`chat` / `api`）、确认状态、更新时间。
4. 支持通过 `PATCH /api/users/{user_id}/memory` 手动添加或修改记忆条目。
5. 支持通过 `DELETE /api/users/{user_id}/memory/{group}/{key}` 删除单条记忆。
6. 对话中也可通过自然语言命令触发：
   - `记住...`：写入偏好
   - `查看长期记忆`：列出当前记忆
   - `删除...`：删除某条记忆

---

## 11. 核心数据流示意

### 11.1 B 站登录流程
```
前端                          后端                        B站
  |                             |                            |
  |-- POST /api/bilibili/auth/qr/start -->|                  |
  |                             |-- 调用 B站生成二维码 API -->|
  |                             |<-- qr_url + qrcode_key ----|  
  |<-- { qr_url, qrcode_key } --|                            |
  | 渲染二维码图片                |                            |
  | 用户用B站App扫码 ------------>|<--- 用户扫码 ------------->|
  |                             |                            |
  |-- GET /api/bilibili/auth/qr/poll?qrcode_key=xxx (轮询) ->|
  |                             |-- 调用 B站轮询 API -------->|
  |                             |<-- status=success, cookie -|
  |<-- { status:success, cookie } --|                        |
  | 保存 cookie 到 localStorage   |                           |
```

### 11.2 导入流程
```
前端                                    后端
  |
  |-- GET /api/bilibili/favorite-folders (带Cookie) -->
  |<-- 收藏夹列表 --
  |
  |-- GET /api/bilibili/favorite-folders/{id}/videos -->
  |<-- 视频列表（含 selectable 标记）--
  |
  | 用户勾选视频
  |
  |-- POST /api/bilibili/imports (带Cookie + selected_video_ids) -->
  |<-- { run_id, status:"running" } -- (HTTP 202)
  |
  |-- GET /api/runs/{run_id}/events (SSE 订阅) -->
  |<-- event: run_started
  |<-- event: tool_executing (每个视频)
  |<-- event: run_completed
  |
  |-- GET /api/runs/{run_id} (最终结果) -->
  |<-- { steps, status:"completed" } --
```

### 11.3 对话 + 确认流程
```
前端                                    后端
  |
  |-- POST /api/chat { message, session_id } -->
  |<-- { status:"awaiting_confirmation", execution_plan, run_id } --
  |
  | 展示执行计划，等待用户确认
  |
  |-- POST /api/runs/{run_id}/confirm { approved: true } -->
  |<-- { status:"completed", reply } --
```

### 11.4 知识问答流程
```
前端                                    后端
  |
  |-- POST /api/chat { message:"这个收藏夹里有关于XX的内容吗？", session_id } -->
  |   (后端自动路由为 favorite_knowledge_query)
  |   (后端调用知识检索 → 知识问答服务)
  |<-- { status:"completed", reply:"根据收藏夹内容...来源: 视频标题", route:"favorite_knowledge_query" } --
```

---

## 附录：前端本地存储约定

| localStorage Key | 类型 | 说明 |
|-----------------|------|------|
| `biliCookie` | string | B 站登录 Cookie，登录成功后写入 |
| `biliAccount` | JSON string | `{ mid, uname }` 账号信息 |
| `sessionId` | string | 当前对话会话 ID |
| `userId` | string | 当前用户 ID（可选，用于长期记忆）|

---

## 附录：接口速查表

| 方法 | 路径 | 说明 | 需要Cookie |
|------|------|------|------------|
| GET | `/health` | 健康检查 | 否 |
| POST | `/api/chat` | 发送消息 | 否 |
| POST | `/api/runs/{run_id}/confirm` | 确认/拒绝执行计划 | 否 |
| GET | `/api/runs/{run_id}` | 获取运行详情 | 否 |
| GET | `/api/runs/{run_id}/events` | SSE 事件流 | 否 |
| GET | `/api/sessions/{session_id}` | 获取会话详情 | 否 |
| GET | `/api/users/{user_id}/memory` | 获取长期记忆 | 否 |
| PATCH | `/api/users/{user_id}/memory` | 更新长期记忆 | 否 |
| DELETE | `/api/users/{user_id}/memory/{group}/{key}` | 删除记忆条目 | 否 |
| POST | `/api/bilibili/auth/qr/start` | 生成登录二维码 | 否 |
| GET | `/api/bilibili/auth/qr/poll` | 轮询登录状态 | 否 |
| GET | `/api/bilibili/favorite-folders` | 获取收藏夹列表 | **是** |
| GET | `/api/bilibili/favorite-folders/{id}/videos` | 获取收藏夹视频 | **是** |
| POST | `/api/bilibili/imports` | 提交导入任务 | **是** |
| POST | `/api/knowledge/search` | 知识库搜索 | 否 |
   