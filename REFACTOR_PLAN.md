# Agent 核心链路改造计划

> 分析日期：2026-03-26  
> 目标：用 LangGraph / LangChain 原生能力替换项目中大量手写的冗余逻辑

---

## 一、现状诊断：哪些代码写得麻烦且多余

### 1. 手写 Event 系统 → 应使用 LangGraph 原生流式输出

**现状代码路径：**
- `app/agent/types.py` — 定义了 14 种 `RunEventType`
- `app/agent/service.py` — 每个节点手动调用 `self._emit_event(run_id, "...", payload)`
- `app/agent/events.py` — `aggregate_chat_response()` 遍历事件列表重建状态（~90行）
- `app/db/repository.py` — `append_run_event` / `get_run_events` / `get_run_event_count`
- `app/api/routes/chat.py` — `stream_run_events()` 手写 SSE 轮询循环（`asyncio.sleep(0.25)`）

**问题：**
- LangGraph `graph.stream()` 原生支持按节点/按 token 流式输出，不需要手写事件表、手写 SSE 轮询
- `aggregate_chat_response` 本质是把事件流还原成状态机——这正是 LangGraph StateGraph 自带的功能
- SSE 轮询用 `asyncio.sleep(0.25)` 做 250ms 轮询，有延迟且浪费 CPU；`graph.astream()` 是真正的 async 推送

---

### 2. 手写 `interrupt` + `confirm` 接口 → 应使用 LangGraph Human-in-the-Loop 原生模式

**现状代码路径：**
- `app/agent/service.py` `_approval_gate` 节点调用 `interrupt(payload)`
- `app/agent/service.py` `resume_run()` 调用 `graph.invoke(Command(resume=...))`
- `app/api/routes/chat.py` `confirm_run()` — 独立 HTTP 接口，手动查 run 状态、手动校验 `awaiting_confirmation`
- `app/db/repository.py` — `update_run` 里维护 `approval_requested_at` / `approval_resolved_at` 等字段

**问题：**
- `interrupt()` + `Command(resume=...)` 是 LangGraph 0.2+ 的标准 Human-in-the-Loop 模式，这部分逻辑本身写对了
- 但 `confirm_run` 接口里大量手动同步 `langsmith_thread_url` / `langsmith_thread_id` 的代码是冗余样板
- `_approval_gate` 节点几乎只是转发数据，可以内联进路由条件，不需要独立节点

---

### 3. 手写 `_detect_route` 关键词路由 → 应使用 LangChain Router / LLM 分类器

**现状代码路径：**
- `app/agent/service.py` `_detect_route()` — 约 30 行关键词 `in lowered` 判断

**问题：**
- 关键词匹配脆弱，容易漏匹配（例如「把这个B站视频的内容告诉我」不含任何关键词）
- LangChain 提供 `ChatPromptTemplate` + structured output（`with_structured_output`）做意图分类，只需定义 `RouteType` 枚举，一次 LLM 调用即可
- 或使用 `RunnableBranch` / `RouterRunnable` 声明式路由

---

### 4. 手写 `runtime_audit` 封装 → 应使用 LangSmith 原生回调

**现状代码路径：**
- `app/services/runtime_audit.py` — 整个文件（`LangSmithRuntimeAudit` + `NoOpRuntimeAudit` + `_NoOpRun`）
- `app/agent/service.py` — 每个节点都有 `with self.runtime_audit.trace_span(...) as trace_run:` + `trace_run.end()` + `trace_run.add_metadata()`

**问题：**
- LangGraph 内置 LangSmith 追踪：只需设置 `LANGSMITH_TRACING=true` 环境变量，所有节点自动上报，不需要在每个节点手动 `trace_span`
- LangChain 的 `CallbackManager` / `@traceable` 装饰器可以替代整个 `runtime_audit` 封装层
- 当前 `_NoOpRun.__getattr__` 的 hack 说明这套封装已经在对抗框架而非利用框架

---

### 5. 无流式 LLM 响应 → 应使用 LangChain `astream` / `StreamingCallbackHandler`

**现状代码路径：**
- `app/services/llm.py` `chat()` — 同步调用 `client.chat.completions.create(...)`，等全部内容返回后一次性返回字符串
- `app/api/routes/chat.py` `chat()` — 同步路由，`POST /api/chat` 无流式输出

**问题：**
- OpenAI SDK 支持 `stream=True`；LangChain `ChatOpenAI` 支持 `astream()` 逐 token 推送
- 前端现在必须等 LLM 全部生成完才能看到回复，用户体验差
- LangGraph `graph.astream(stream_mode="messages")` 可以在图执行过程中逐 token 流式推送

---

### 6. 手写 `_NoOpRun` 应对本地无 LangSmith 的情况 → 应用环境变量控制

**现状：** 用 `NoOpRuntimeAudit` + `_NoOpRun.__getattr__` hack 在本地跳过追踪  
**正确方式：** 设置 `LANGSMITH_TRACING=false`，LangGraph/LangChain 会自动跳过追踪，完全不需要 NoOp 封装层

---

## 二、改造计划（优先级排序）

### P0 — 流式 LLM 响应（影响用户体验，立刻可做）

**目标：** `POST /api/chat` 改为 SSE 流式返回 LLM token

**方案：**
1. 将 `OpenAICompatibleLLM.chat()` 改用 LangChain `ChatOpenAI`（支持 `astream()`）
2. 新增 `POST /api/chat/stream` 接口，返回 `StreamingResponse`
3. 逐 token `yield f"data: {token}\n\n"`
4. 流式结束后追加一条 `data: [DONE]\n\n`
5. 前端 `ChatPage` 改用 `EventSource` 或 `fetch` with `ReadableStream` 接收

**涉及文件：**
- `app/services/llm.py`
- `app/api/routes/chat.py`
- `frontend/src/api/agent.ts`
- `frontend/src/pages/ChatPage.tsx`

**预计工作量：** 1-2 天

---

### P1 — 用 `graph.astream()` 替换手写 Event + SSE 轮询

**目标：** 删除 `app/agent/events.py`、`run_events` 表、SSE 轮询循环

**方案：**
1. 将 `graph.invoke()` 改为 `graph.astream(stream_mode=["updates", "messages"])`
2. `POST /api/chat` 直接返回 `StreamingResponse`，每个节点完成时推送一条 SSE 事件
3. `stream_mode="messages"` 模式下，LLM 节点会逐 token 推送
4. `stream_mode="updates"` 模式下，每个节点完成时推送状态 diff
5. 前端的 `GET /api/runs/{run_id}/events` SSE 接口可以废弃，改用 chat 接口本身的流

```python
# 改造前
result = self.graph.invoke(state, config=config)

# 改造后
async for chunk in self.graph.astream(state, config=config, stream_mode="messages"):
    node_name, message_chunk = chunk
    yield f"data: {message_chunk.content}\n\n"
```

**涉及文件：**
- `app/agent/service.py` — 删除所有 `_emit_event` 调用
- `app/agent/events.py` — **整个文件可删除**
- `app/api/routes/chat.py` — 删除 `stream_run_events` 接口，chat 接口改 async
- `app/db/repository.py` — 删除 `run_events` 相关方法

**预计工作量：** 2-3 天

---

### P2 — 移除手写 `runtime_audit`，改用 LangSmith 原生追踪

**目标：** 删除 `app/services/runtime_audit.py`，每个节点不再有手动 `trace_span`

**方案：**
1. 在 `.env` 中设置 `LANGSMITH_TRACING=true/false`（已有）
2. LangGraph 会自动把每个节点作为一个 span 上报到 LangSmith
3. 删除 `app/agent/service.py` 中所有 `with self.runtime_audit.trace_span(...)` 代码块
4. 节点函数回归纯粹的业务逻辑，代码量减少约 40%
5. 对需要自定义 metadata 的地方用 `@traceable` 装饰器或 `langsmith.RunTree` 即可

**涉及文件：**
- `app/services/runtime_audit.py` — **整个文件可删除**
- `app/agent/service.py` — 删除所有 trace_span 包裹
- `app/main.py` — 删除 `runtime_audit` 初始化和注入

**预计工作量：** 1 天

---

### P3 — 用 LLM 结构化输出替换关键词路由

**目标：** 替换 `_detect_route()` 关键词匹配

**方案：**
```python
# 改造前
def _detect_route(self, message: str) -> RouteType:
    lowered = message.lower()
    if any(kw in lowered for kw in retry_keywords):
        return "retry_request"
    ...

# 改造后：用 ChatOpenAI + with_structured_output
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

class RouteDecision(BaseModel):
    route: RouteType
    reason: str

router_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "你是一个意图分类器，根据用户消息返回 route 类型..."),
        ("human", "{message}"),
    ])
    | ChatOpenAI(model="...").with_structured_output(RouteDecision)
)

route_decision = router_chain.invoke({"message": state["current_message"]})
```

**涉及文件：**
- `app/agent/service.py` — 替换 `_detect_route` 方法

**注意：** 这会增加一次 LLM 调用延迟，可以用缓存（`langchain.cache`）或在 P0 流式之后做

**预计工作量：** 0.5 天

---

### P4 — 简化 `approval_gate` 节点

**目标：** 把 `approval_gate` 从独立节点内联为路由条件

**方案：**
- 当前 `approval_gate` 节点的唯一职责是调用 `interrupt()` 然后判断用户决策
- 可以将其合并到 `_route_after_plan` 路由函数中，减少一个图节点
- LangGraph 0.3+ 的 `interrupt()` 可以直接在条件边里调用

**涉及文件：**
- `app/agent/service.py`

**预计工作量：** 0.5 天

---

## 三、改造路线图

```
第一阶段（本周）
├── P0: 流式 LLM 响应
│   ├── llm.py 改用 ChatOpenAI + astream
│   └── /api/chat/stream 新接口
│
第二阶段（下周）
├── P1: graph.astream 替换手写 Event
│   ├── 删除 events.py
│   ├── 删除 run_events 表
│   └── chat 接口改 StreamingResponse
│
第三阶段
├── P2: 移除 runtime_audit 手写封装
│   └── 改用 LANGSMITH_TRACING 环境变量
│
第四阶段（可选）
├── P3: LLM 结构化路由
└── P4: 简化 approval_gate
```

---

## 四、改造后的代码量对比

| 模块 | 改造前 | 改造后 | 节省 |
|------|--------|--------|------|
| `app/agent/events.py` | ~90 行 | 0 行（删除） | 100% |
| `app/services/runtime_audit.py` | ~150 行 | 0 行（删除） | 100% |
| `app/agent/service.py` trace_span 样板 | ~200 行 | ~20 行 | 90% |
| `app/api/routes/chat.py` SSE 轮询 | ~35 行 | ~10 行 | 70% |
| `app/db/repository.py` run_events 方法 | ~40 行 | 0 行（删除） | 100% |
| **合计** | **~515 行** | **~30 行** | **~94%** |

---

## 五、改造不动的部分

以下代码写得正确，不需要改造：

- `interrupt()` + `Command(resume=...)` 的 Human-in-the-Loop 模式 ✅
- `SqliteSaver` checkpoint 持久化 ✅  
- `KnowledgeRetrievalService` + ChromaDB 检索 ✅
- `UserMemoryManager` 长期记忆管理 ✅
- B站 API 封装 (`BilibiliFavoriteFolderService`) ✅
- 各页面路由结构 (`StateGraph` 节点定义) ✅

---

## 六、参考文档

- [LangGraph Streaming](https://langchain-ai.github.io/langgraph/how-tos/streaming/)
- [LangGraph Human-in-the-Loop](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/)
- [LangSmith 自动追踪（无需手写）](https://docs.smith.langchain.com/observability/how_to_guides/trace_with_langgraph)
- [ChatOpenAI streaming](https://python.langchain.com/docs/how_to/chat_streaming/)
- [with_structured_output](https://python.langchain.com/docs/how_to/structured_output/)
