import { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { sendChat, confirmRun } from '../api/agent';
import './ChatPage.css';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  route?: string;
  runId?: string;
  status?: string;
  executionPlan?: any;
  requiresConfirmation?: boolean;
  approvalStatus?: string;
}

const routeLabel: Record<string, string> = {
  general_chat: '通用对话',
  favorite_knowledge_query: '收藏夹问答',
  video_knowledge_query: '视频问答',
  import_request: '导入任务',
  retry_request: '重试任务',
};
const routeTagClass: Record<string, string> = {
  general_chat: 'tag-gray',
  favorite_knowledge_query: 'tag-orange',
  video_knowledge_query: 'tag-blue',
  import_request: 'tag-green',
  retry_request: 'tag-yellow',
};

function genId() { return Math.random().toString(36).slice(2); }

export default function ChatPage() {
  const navigate = useNavigate();
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [confirming, setConfirming] = useState<Record<string, boolean>>({});
  const bottomRef = useRef<HTMLDivElement>(null);

  const sessionId = useRef<string>(
    localStorage.getItem('sessionId') ?? (() => {
      const id = genId();
      localStorage.setItem('sessionId', id);
      return id;
    })()
  );
  const userId = localStorage.getItem('userId') ?? undefined;

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const addMsg = (m: Message) => setMessages(prev => [...prev, m]);

  const handleSend = async () => {
    const text = input.trim();
    if (!text || loading) return;
    setInput('');
    addMsg({ id: genId(), role: 'user', content: text });
    setLoading(true);
    try {
      const res = await sendChat({ session_id: sessionId.current, user_id: userId, message: text });
      sessionId.current = res.session_id;
      localStorage.setItem('sessionId', res.session_id);
      addMsg({
        id: genId(),
        role: 'assistant',
        content: res.reply,
        route: res.route ?? undefined,
        runId: res.run_id,
        status: res.status,
        executionPlan: res.execution_plan,
        requiresConfirmation: res.requires_confirmation,
        approvalStatus: res.approval_status,
      });
    } catch (e: any) {
      addMsg({ id: genId(), role: 'assistant', content: `❌ ${e?.response?.data?.detail ?? '请求失败'}` });
    } finally {
      setLoading(false);
    }
  };

  const handleConfirm = async (runId: string, approved: boolean) => {
    setConfirming(prev => ({ ...prev, [runId]: true }));
    try {
      const res = await confirmRun(runId, approved);
      setMessages(prev => prev.map(m =>
        m.runId === runId
          ? { ...m, status: res.status, approvalStatus: res.approval_status, requiresConfirmation: false, content: m.content }
          : m
      ));
      addMsg({
        id: genId(),
        role: 'assistant',
        content: res.reply,
        route: res.route ?? undefined,
        runId: res.run_id,
        status: res.status,
      });
    } catch (e: any) {
      addMsg({ id: genId(), role: 'assistant', content: `❌ ${e?.response?.data?.detail ?? '确认失败'}` });
    } finally {
      setConfirming(prev => ({ ...prev, [runId]: false }));
    }
  };

  const newSession = () => {
    const id = genId();
    sessionId.current = id;
    localStorage.setItem('sessionId', id);
    setMessages([]);
  };

  return (
    <div className="chat-page">
      <div className="chat-topbar">
        <span className="chat-title">对话</span>
        <div className="chat-topbar-actions">
          <button className="btn btn-ghost" onClick={newSession}>新会话</button>
          <button className="btn btn-ghost" onClick={() => navigate('/memory')}>记忆管理</button>
        </div>
      </div>

      <div className="chat-messages">
        {messages.length === 0 && (
          <div className="chat-empty">
            <div className="chat-empty-icon">◎</div>
            <p>向 Agent 提问，或询问你的 B 站收藏夹内容</p>
            <div className="chat-hints">
              {['我的收藏夹里有什么关于 React 的视频？', '这个视频讲了什么？', '帮我导入收藏夹'].map(h => (
                <button key={h} className="hint-chip" onClick={() => { setInput(h); }}>{h}</button>
              ))}
            </div>
          </div>
        )}

        {messages.map((m) => (
          <div key={m.id} className={`msg-row ${m.role} fade-in`}>
            <div className="msg-bubble">
              {m.role === 'assistant' && m.route && (
                <span className={`tag ${routeTagClass[m.route] ?? 'tag-gray'} msg-route-tag`}>
                  {routeLabel[m.route] ?? m.route}
                </span>
              )}
              <div className="msg-content">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.content}</ReactMarkdown>
              </div>

              {m.requiresConfirmation && m.executionPlan && m.runId && (
                <div className="execution-plan">
                  <p className="plan-goal">🎯 {m.executionPlan.goal}</p>
                  <p className="plan-summary">{m.executionPlan.summary}</p>
                  {m.executionPlan.steps?.length > 0 && (
                    <ul className="plan-steps">
                      {m.executionPlan.steps.map((s: any) => (
                        <li key={s.id}><strong>{s.title}</strong> — {s.description}</li>
                      ))}
                    </ul>
                  )}
                  <div className="plan-actions">
                    <button
                      className="btn btn-success"
                      disabled={!!confirming[m.runId]}
                      onClick={() => handleConfirm(m.runId!, true)}
                    >
                      {confirming[m.runId] ? <><div className="spinner" style={{width:14,height:14}}/> 处理中</> : '✓ 确认执行'}
                    </button>
                    <button
                      className="btn btn-danger"
                      disabled={!!confirming[m.runId]}
                      onClick={() => handleConfirm(m.runId!, false)}
                    >✕ 取消</button>
                  </div>
                </div>
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="msg-row assistant">
            <div className="msg-bubble typing">
              <span /><span /><span />
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <div className="chat-input-bar">
        <textarea
          className="chat-textarea"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); } }}
          placeholder="输入消息，Enter 发送，Shift+Enter 换行..."
          rows={2}
        />
        <button className="btn btn-primary chat-send-btn" onClick={handleSend} disabled={loading || !input.trim()}>
          {loading ? <div className="spinner" style={{width:16,height:16}} /> : '发送'}
        </button>
      </div>
    </div>
  );
}
