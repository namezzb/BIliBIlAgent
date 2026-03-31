import pytest
from unittest.mock import MagicMock


def _make_orchestrator_double():
    from app.agent.service import AgentOrchestrator

    orch = MagicMock(spec=AgentOrchestrator)
    orch.llm = MagicMock()
    orch.llm.api_key = None
    orch.repository = MagicMock()
    orch.repository.upsert_run_step.return_value = None
    orch.repository.list_knowledge_videos.return_value = [
        {"video_id": "test_embed_001", "bvid": "BVtest001", "title": "Transformer注意力机制详解"},
        {"video_id": "test_embed_002", "bvid": "BVtest002", "title": "React源码解析"},
    ]
    orch.repository.list_knowledge_favorite_folders.return_value = [
        {"favorite_folder_id": "986563070", "title": "默认收藏夹"},
        {"favorite_folder_id": "123", "title": "React精选"},
    ]

    # Bind real methods used by router flow.
    orch._router = AgentOrchestrator._router.__get__(orch)
    orch._detect_intent = AgentOrchestrator._detect_intent.__get__(orch)
    orch._detect_knowledge_scope = AgentOrchestrator._detect_knowledge_scope.__get__(orch)
    orch._detect_action_type = AgentOrchestrator._detect_action_type.__get__(orch)
    orch._keyword_detect_intent = AgentOrchestrator._keyword_detect_intent.__get__(orch)
    orch._keyword_detect_knowledge_scope = AgentOrchestrator._keyword_detect_knowledge_scope.__get__(orch)
    orch._keyword_detect_action_type = AgentOrchestrator._keyword_detect_action_type.__get__(orch)
    orch._looks_like_action_request = AgentOrchestrator._looks_like_action_request.__get__(orch)
    orch._looks_like_knowledge_query = AgentOrchestrator._looks_like_knowledge_query.__get__(orch)
    orch._looks_like_video_scope = AgentOrchestrator._looks_like_video_scope.__get__(orch)
    orch._looks_like_folder_scope = AgentOrchestrator._looks_like_folder_scope.__get__(orch)
    orch._extract_bvids = AgentOrchestrator._extract_bvids.__get__(orch)
    orch._extract_explicit_video_titles = AgentOrchestrator._extract_explicit_video_titles.__get__(orch)
    orch._extract_explicit_folder_names = AgentOrchestrator._extract_explicit_folder_names.__get__(orch)
    return orch


def _run_router(message: str):
    orch = _make_orchestrator_double()
    state = {"run_id": "run-1", "current_message": message}
    return orch._router(state)


def test_router_chat_intent_routes_to_general_chat():
    result = _run_router("你好")
    assert result["intent"] == "chat"
    assert result["route"] == "general_chat"
    assert result["retrieval_scope_hint"] is None


@pytest.mark.parametrize("message", ["注意力机制是什么？", "GIL 是什么？", "自注意力和交叉注意力有什么区别？"])
def test_router_concept_questions_route_to_general_knowledge(message: str):
    result = _run_router(message)
    assert result["intent"] == "knowledge"
    assert result["route"] == "knowledge_query"
    assert result["retrieval_scope_hint"] == "general_knowledge_query"


def test_router_bvid_question_routes_to_video_scope():
    result = _run_router("BVtest001 这个视频讲了什么？")
    assert result["intent"] == "knowledge"
    assert result["route"] == "knowledge_query"
    assert result["retrieval_scope_hint"] == "video_knowledge_query"
    assert result["mentioned_bvids"] == ["BVTEST001"]



def test_router_title_question_routes_to_video_scope():
    result = _run_router("Transformer注意力机制详解 这个视频讲了什么？")
    assert result["intent"] == "knowledge"
    assert result["retrieval_scope_hint"] == "video_knowledge_query"
    assert result["mentioned_video_titles"] == ["Transformer注意力机制详解"]



def test_router_folder_question_routes_to_folder_scope():
    result = _run_router("默认收藏夹里有哪些 React 视频？")
    assert result["intent"] == "knowledge"
    assert result["route"] == "knowledge_query"
    assert result["retrieval_scope_hint"] == "favorite_knowledge_query"
    assert result["mentioned_folder_names"] == ["默认收藏夹"]



def test_router_followup_video_question_routes_to_video_scope():
    result = _run_router("这个视频还讲了什么？")
    assert result["intent"] == "knowledge"
    assert result["retrieval_scope_hint"] == "video_knowledge_query"



def test_router_import_request_routes_to_action_plan():
    result = _run_router("帮我导入收藏夹")
    assert result["intent"] == "action"
    assert result["action_route"] == "import_request"
    assert result["route"] == "plan_and_solve"



def test_router_retry_request_routes_to_action_plan():
    result = _run_router("帮我重试上次失败的导入任务")
    assert result["intent"] == "action"
    assert result["action_route"] == "retry_request"
    assert result["route"] == "plan_and_solve"
