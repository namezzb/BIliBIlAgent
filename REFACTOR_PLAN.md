# Agent 核心链路改造计划

> 分析日期：2026-03-26（官方文档验证修订版）  
> 验证来源：Context7 LangGraph Python 官方文档 + LangSmith 官方文档 + Web Search

---

## 一、现状诊断：哪些代码写得麻烦且多余

### 1. 手写 Event 系统 → 应使用 `graph.astream()` 原生流式

**现状代码路径：**
- `app/agent/types.py` — 定义 14 种 `RunEventType`
- `app/agent/service.py` — 每个节点手动调用 `self._emit_event(run_id, "...", payload)`
- `app/agent/events.py` — `aggregate_chat_response()` 遍历事件列表重建状态（~90行）
- `app/db/repository.py` — `append_run_event` / `get_run_events` 方法
- `app/api/routes/chat.py` — `stream_run_events()` 手写 250ms 轮询 SSE 循环

**官方文档确认：**  
LangGraph 原生支持 `graph.astream(stream_mode=["messages", "updates"])` + `version="v2"`，每个节点完成时自动推送 `updates` chunk，LLM 节点逐 token 推送 `messages` chunk。**不需要任何手写事件表或轮询。**

```python
# 官方标准用法（已验证）
async for chunk in graph.astream(
    initial_input,
    stream_mode=["messages", "updates"],
    version="v2",
    config=config,
):
    if chunk["type"] == "messages":
        msg, metadata = chunk["data"]
        # msg 是 AIMessageChunk，逐 token 推送
    elif chunk["type"] == "updates":
        # 每个节点完成后的状态 diff
        for node_name, state_diff in chunk["data"].items():
            pass  # 可以推给前端作为进度
```

---

### 2. SSE 轮询实现有根本性缺陷

**现状：** `asyncio.sleep(0.25)` 250ms 轮询数据库，最坏情况下延迟 250ms，且持续占用数据库连接。

**正确方案：** `chat` 接口本身改为 `StreamingResponse`，直接把 `graph.astream()` 的 chunk 通过 SSE 推给前端，**零轮询、零延迟。**

```python
# FastAPI + LangGraph astream 正确集成方式（已验证）
@router.post("/api/chat/stream")
async def chat_stream(request: Request, payload: ChatRequest):
    async def event_generator():
        async for chunk in graph.astream(
            state, config=config,
            stream_mode=["messages", "updates"],
            version="v2",
        ):
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

---

### 3. interrupt + confirm 接口逻辑正确，但有冗余同步代码

**现状：** `interrupt()` + `Command(resume=...)` 用法**完全符合官方规范**，不需要改。  
但 `confirm_run` 接口里手动同步 `langsmith_thread_url` / `langsmith_thread_id` 的代码是冗余样板。

**官方文档确认的标准用法：**

```python
# 官方：interrupt 在节点内直接调用（已验证）
def approval_gate(state: AgentState):
    decision = interrupt({"question": "批准执行？", "plan": state["execution_plan"]})
    return {"approval_status": "approved" if decision["approved"] else "rejected"}

# 官方：resume 用 Command（已验证）
result = graph.invoke(
    Command(resume={"approved": True}),
    config={"configurable": {"thread_id": run_id}}
)
```

**新发现（重要）：** 在流式场景下，`interrupt` 触发时会在 `updates` chunk 里出现 `__interrupt__` 键，前端可以直接从 SSE 流里检测到，**不需要轮询 `/runs/{run_id}` 接口**：

```python
# 流式场景下检测 interrupt（官方文档已验证）
elif chunk["type"] == "updates":
    if "__interrupt__" in chunk["data"]:
        interrupt_info = chunk["data"]["__interrupt__"][0].value
        # 把 interrupt_info 推给前端显示确认弹窗
```

---

### 4. 手写 `runtime_audit` 封装 → 只需环境变量

**现状：** `app/services/runtime_audit.py` 整个文件（`LangSmithRuntimeAudit` + `NoOpRuntimeAudit` + `_NoOpRun`），每个节点都有手动 `trace_span` 包裹。

**官方文档确认：**  
LangGraph 只需设置两个环境变量即可自动追踪所有节点，**零代码**：

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=<your-api-key>
```

本地开发：`LANGSMITH_TRACING=false`，LangGraph 自动跳过追踪，完全不需要 `_NoOpRun` hack。

**⚠️ 修正原计划的一处错误：**  
原计划说「`@traceable` 装饰器可以替代整个 runtime_audit」。实际上 `@traceable` 是 LangSmith SDK 的装饰器，用于**非 LangGraph** 的普通 Python 函数。在 LangGraph 图节点里，**不需要 `@traceable`**，因为图本身已经自动追踪每个节点。只有图外的工具函数（如 `BilibiliFavoriteFolderService`）才需要 `@traceable`。

---

### 5. 无流式 LLM 响应

**现状：** `llm.py` 同步调用，等全部内容返回才响应前端。

**官方文档确认的方案：**  
使用 `init_chat_model()` 或 `ChatOpenAI()`，在图节点内用 `.invoke()`（**不是 `.stream()`**）调用 LLM——LangGraph 的 `stream_mode="messages"` 会自动拦截并逐 token 推送，**节点内部不需要手动 stream**：

```python
# 官方推荐：节点内用 .invoke()，graph 层面用 astream
from langchain.chat_models import init_chat_model

model = init_chat_model(model="qwen/qwen-turbo", base_url="...", api_key="...")

def plan_or_answer(state: AgentState):
    # 直接用 invoke，LangGraph 自动 stream token
    response = model.invoke(state["messages"])
    return {"response": response.content}
```

---

### 6. 关键词路由脆弱（原有分析正确）

**现状：** `_detect_route()` 硬编码关键词，容易漏匹配自然语言。

**官方支持的方案：**

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

class RouteDecision(BaseModel):
    route: RouteType
    reason: str

router_llm = ChatOpenAI(model="...").with_structured_output(RouteDecision)
```

**注意：** `with_structured_output` 在 OpenAI 兼容接口（如 OpenRouter）上可用，但需确认上游模型支持 function calling / tool use。

---

### 7. 【新增遗漏项】`graph.astream` 与 confirm 接口的架构冲突

**这是一个重要的设计问题，原计划未提及。**

当 `POST /api/chat` 改为流式 SSE 时，`interrupt` 的处理方式需要重新设计：

**当前架构（非流式）：**
```
前端 POST /api/chat → 后端等待 → 返回 {requires_confirmation: true}
前端 POST /api/runs/{run_id}/confirm → 后端 resume → 返回结果
```

**流式架构（两种选择）：**

**方案 A：Chat 接口 SSE + 独立 Confirm 接口（推荐，改动最小）**
```
前端 POST /api/chat/stream → SSE 流
  ├── chunk: messages token...
  ├── chunk: updates {__interrupt__: {question, plan}}  ← 前端检测到，弹出确认框
  └── SSE 流暂停（图被 interrupt 挂起）
前端 POST /api/runs/{run_id}/confirm → 后端 resume → 新的 SSE 流推送结果
```

**方案 B：单 WebSocket 连接（改动最大）**
```
前端 WS /api/chat/ws → 全程保持连接
  前端发送消息 → 后端 astream → 逐 token 推送
  interrupt → 后端推送确认请求 → 前端回复 approved=true → 后端 resume → 继续推送
```

**结论：推荐方案 A**，与现有 REST API 兼容，改动最小。前端只需监听 SSE 流里的 `__interrupt__` 类型事件弹出确认框，点击后调用现有 `POST /api/runs/{run_id}/confirm`，confirm 接口也改为 SSE 流式返回。

---

### 8. 【新增遗漏项】`SqliteSaver` 需升级为异步版本

当 `graph.invoke()` 改为 `graph.astream()` 后，FastAPI 路由变成 `async def`。  
当前 `SqliteSaver` 是同步版本，在 async 上下文中会阻塞事件循环。

```python
# 当前（同步，会阻塞）
from langgraph.checkpoint.sqlite import SqliteSaver

# 改造后（异步）
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
# 需要：poetry add aiosqlite
```

---

### 9. 【新增遗漏项】前端需从 `EventSource` 改为 `fetch` + `ReadableStream`

浏览器原生 `EventSource` 只支持 `GET`，不支持 `POST` body。  
`POST /api/chat/stream` 必须用 `fetch` + `ReadableStream`：

```typescript
const response = await fetch('/api/chat/stream', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ message, session_id }),
});
const reader = response.body!.getReader();
const decoder = new TextDecoder();
while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  for (const line of decoder.decode(value).split('\n')) {
    if (line.startsWith('data: ') && line !== 'data: [DONE]') {
      const chunk = JSON.parse(line.slice(6));
      if (chunk.type === 'token') appendToken(chunk.content);
      if (chunk.type === 'interrupt') showConfirmDialog(chunk.data);
    }
  }
}
```

---

## 二、改造路线图

```
第一阶段（本周，优先）
├── P0: 流式 LLM 响应
│   ├── llm.py 改用 init_chat_model / ChatOpenAI
│   ├── 新增 POST /api/chat/stream（astream + StreamingResponse）
│   ├── poetry add aiosqlite → 换 AsyncSqliteSaver
│   └── 前端 ChatPage 改用 fetch + ReadableStream
│
第二阶段（下周）
└── P1: 删除手写 Event 系统
    ├── 删除 events.py + run_events 表
    ├── astream updates 中检测 __interrupt__
    └── 废弃 GET /api/runs/{run_id}/events 轮询接口

第三阶段
└── P2: 移除 runtime_audit 封装
    └── 环境变量 LANGSMITH_TRACING + @traceable 装饰器

第四阶段（可选）
├── P3: LLM 结构化路由（with_structured_output）
└── P4: 简化 approval_gate 节点
```

---

## 三、改造后代码量对比

| 模块 | 改造前 | 改造后 | 节省 |
|------|--------|--------|------|
| `app/agent/events.py` | ~90 行 | 0（删除） | 100% |
| `app/services/runtime_audit.py` | ~150 行 | 0（删除） | 100% |
| `app/agent/service.py` trace_span 样板 | ~200 行 | ~20 行 | 90% |
| `app/api/routes/chat.py` SSE 轮询 | ~35 行 | ~15 行 | 57% |
| `app/db/repository.py` run_events 方法 | ~40 行 | 0（删除） | 100% |
| **合计** | **~515 行** | **~35 行** | **~93%** |

---

## 四、原计划的错误修正（官方文档确认）

| # | 原计划说法 | 修正 |
|---|-----------|------|
| 1 | `@traceable` 可替代整个 runtime_audit | 错误。`@traceable` 用于图**外**的普通函数；图节点只需环境变量，无需任何装饰器 |
| 2 | 节点内 LLM 要改为 `.stream()` | 错误。节点内继续用 `.invoke()`；`graph.astream(stream_mode="messages")` 会自动拦截并推送 token |
| 3 | `stream_mode` 直接传字符串 | 需补充：多模式时传列表 `["messages", "updates"]`，并加 `version="v2"` 才能得到统一的 `StreamPart` 格式 |
| 4 | interrupt 只能用 confirm 接口检测 | 补充：流式场景下 `updates` chunk 里会出现 `__interrupt__` 键，前端可直接从 SSE 流里检测 |

---

## 五、改造不动的部分

- `interrupt()` + `Command(resume=...)` 的 Human-in-the-Loop 模式 ✅
- `StateGraph` 节点定义结构 ✅
- `KnowledgeRetrievalService` + ChromaDB 检索 ✅
- `UserMemoryManager` 长期记忆管理 ✅
- B站 API 封装 (`BilibiliFavoriteFolderService`) ✅
- `POST /api/runs/{run_id}/confirm` 接口（仅内部改为 astream）✅

---

## 六、参考文档

- [LangGraph Streaming（stream_mode, version=v2）](https://docs.langchain.com/oss/python/langgraph/streaming)
- [LangGraph Human-in-the-Loop + 流式 interrupt 检测](https://docs.langchain.com/oss/python/langgraph/interrupts)
- [LangSmith 自动追踪（无需手写）](https://docs.langchain.com/oss/python/langgraph/observability)
- [ChatOpenAI with_structured_output](https://docs.langchain.com/oss/python/integrations/chat/edenai)
- [FastAPI + LangGraph SSE 集成](https://dev.to/kasi_viswanath/streaming-ai-agent-with-fastapi-langgraph-2025-26-guide-1nkn)
 