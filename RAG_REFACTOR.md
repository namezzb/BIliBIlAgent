# RAG 流程改造计划

## 背景与核心问题

当前 RAG 实现几乎没有利用 LangChain/LangGraph 的框架能力，大量逻辑是手工实现的，导致：
- 向量检索距离公式错误（L2 距离误用为 cosine）
- 多轮对话上下文继承完全失效
- Tool 包装层制造了双路径数据断层
- LCEL chain 没有被使用，无法利用 LangGraph 原生 token 流

---

## 改造项总览

| # | 改造项 | 涉及文件 | 优先级 |
|---|---|---|---|
| 1 | 用 `langchain-chroma` 替换手写 `ChromaVectorIndex`，修复 cosine 距离 | `knowledge_index.py`, `main.py`, `pyproject.toml` | 最高 |
| 2 | `_retrieve_knowledge` 节点直接调 service，去掉 Tool 包装层 | `agent/service.py`, `main.py`, `agent/tools.py` | 高 |
| 3 | `KnowledgeGroundedQAService` 改用 LCEL chain，支持 LangGraph token 流 | `knowledge_qa.py` | 高 |
| 4 | `AgentState.retrieval_result` 加 reducer，`finalize_run` 带出，修复多轮上下文断链 | `agent/types.py`, `agent/service.py`, `api/routes/chat.py` | 高 |
| 5 | `RouteDecision` 扩展 LLM 范围识别字段，替换正则子串匹配 | `agent/types.py`, `agent/service.py`, `knowledge_retrieval.py` | 中 |

---

## 改造一：用 `langchain-chroma` 替换手写 `ChromaVectorIndex`

### 问题
- 手写 `ChromaVectorIndex` 协议类 + 实现，重复造轮子
- 使用默认 L2 距离，`score = 1.0 - distance` 公式语义错误（L2 distance ∈ [0,∞)，不适合直接做相似度）
- `embed_texts` 回调函数手工传入，与 LangChain Embeddings 接口割裂

### 方案
1. 安装 `langchain-chroma`
2. `KnowledgeIndexService` 内部用 `Chroma` VectorStore，指定 `hnsw:space=cosine`
3. `index_documents` 用 `vector_store.add_documents()` 替代手写 `upsert`
4. `search` 用 `vector_store.similarity_search_with_relevance_scores()` 替代手写 `query`
5. 删除 `ChromaVectorIndex`、`VectorIndex` Protocol 类
6. `main.py` 构造 `KnowledgeIndexService` 时传入 `lc_embeddings`（`langchain-openai` 的 `OpenAIEmbeddings` 或自定义封装）

### 关键代码变化

```python
# knowledge_index.py
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

class KnowledgeIndexService:
    def __init__(self, repository, *, lc_embeddings: Embeddings,
                 persist_dir, collection_name,
                 embedding_model, embedding_version,
                 chunk_size, chunk_overlap):
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=lc_embeddings,
            persist_directory=str(persist_dir),
            collection_metadata={"hnsw:space": "cosine"},
        )

    def search(self, payload):
        results = self.vector_store.similarity_search_with_relevance_scores(
            query, k=candidate_count
        )
        # results: list[(Document, float)]，score 已是 [0,1] cosine 相似度
```

### 注意
- 已有的 ChromaDB collection 用的是默认 L2，需要删除旧 collection 或换 collection 名称重建
- `pyproject.toml` 添加 `langchain-chroma>=0.1.0`

---

## 改造二：`_retrieve_knowledge` 节点去掉 Tool 包装层

### 问题
- `knowledge_retrieval_tool.invoke()` 返回 `ToolMessage`，hits 从 `.artifact` 取，context 从 `.content` 取，两条路径
- 如果 `artifact` 为空（异常情况），hits 为空但 content 有内容，QA 直接返回"暂无内容"
- Tool 包装层没有任何额外价值（不需要 LLM tool calling，只是手动 invoke）

### 方案
- 节点直接调 `knowledge_retrieval_service.retrieve_for_question()`
- 结果直接写入 `AgentState["retrieval_result"]`，单一来源
- 删除 `knowledge_retrieval_tool` 注入和 `build_knowledge_retrieval_tool`

```python
# agent/service.py
def _retrieve_knowledge(self, state: AgentState) -> dict[str, Any]:
    scope_hint = state.get("retrieval_scope_hint") or "general_knowledge_query"
    retrieval_result = self.knowledge_retrieval_service.retrieve_for_question(
        message=state["current_message"],
        route=scope_hint,
        recent_context=state.get("recent_context", {}),
        top_k=5,
    )
    return {"retrieval_result": retrieval_result}
```

---

## 改造三：`KnowledgeGroundedQAService` 改用 LCEL chain

### 问题
- 手写 `prompt.format_messages()` → `llm.chat()` → 手动字符串处理
- 不是 LCEL chain，LangGraph `stream_mode="messages"` 无法捕获 token 流
- 前端看不到逐字输出

### 方案
- 用 `prompt | lc_llm | StrOutputParser()` 构建标准 LCEL chain
- `answer()` 调用 `self.chain.invoke()`，LangGraph 自动捕获 token 流

```python
# knowledge_qa.py
from langchain_core.output_parsers import StrOutputParser

class KnowledgeGroundedQAService:
    def __init__(self, llm: OpenAICompatibleLLM):
        self.lc_llm = llm.get_lc_chat_model()
        self.chain = self.prompt | self.lc_llm | StrOutputParser()

    def answer(self, *, question, retrieval_result):
        hits = retrieval_result.get("hits", [])
        if not hits:
            return "知识库暂无相关内容。"
        context = retrieval_result.get("serialized_context", "")
        return self.chain.invoke({"question": question, "context": context})
```

---

## 改造四：`AgentState.retrieval_result` 加 reducer，修复多轮上下文断链

### 问题
- `_retrieve_knowledge` 把结果写入 state，但 `finalize_run` 没有输出 `retrieval_result`
- `done_payload` 里没有 `retrieval_result`
- `chat.py` 里 `refresh_session_memory(retrieval_result=None)` 硬编码
- 导致 `recent_context["last_retrieval"]` 永远为空，多轮继承完全失效

### 方案
1. `agent/types.py`：给 `retrieval_result` 加 reducer，保留最新非 None 值
2. `agent/service.py`：`finalize_run` 输出 `retrieval_result`；`astream_chat` 的 done yield 带出
3. `api/routes/chat.py`：`refresh_session_memory` 传入 `done_payload.get("retrieval_result")`
4. `confirm/stream` 路径同步修复

```python
# agent/types.py
def _keep_latest(old, new):
    return new if new is not None else old

class AgentState(TypedDict, total=False):
    retrieval_result: Annotated[dict[str, Any] | None, _keep_latest]
```

---

## 改造五：`RouteDecision` 扩展，范围识别交给 LLM Router

### 问题
- `_resolve_scope` 用 `title in lowered` 子串匹配，短标题极易误匹配
- 正则匹配 BV/AV 号虽准确，但视频标题/收藏夹名识别脆弱

### 方案
- 扩展 `RouteDecision` 增加 `mentioned_bvids`、`mentioned_video_titles`、`mentioned_folder_names` 字段
- LLM 路由时顺便识别用户提及的视频/收藏夹
- `_resolve_scope` 优先使用 LLM 识别结果，正则作为兜底

```python
# agent/types.py
class RouteDecision(BaseModel):
    route: _LLMRouteType
    reason: str
    mentioned_bvids: list[str] = Field(default_factory=list)
    mentioned_video_titles: list[str] = Field(default_factory=list)
    mentioned_folder_names: list[str] = Field(default_factory=list)
```

---

## 执行顺序

```
改造一（langchain-chroma）
  → 改造三（LCEL chain，依赖 lc_llm）
  → 改造二（去掉 Tool 包装，依赖改造一的 service 接口稳定）
  → 改造四（reducer + finalize_run，修复上下文断链）
  → 改造五（LLM 范围识别，依赖改造四完成）
```
