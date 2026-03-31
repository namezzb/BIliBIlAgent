import { useState, useRef, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { streamChat, streamConfirm, type StreamChunk } from '../api/agent';
import './ChatPage.css';

interface AgentStep {
  node: string;
  status: 'running' | 'done';
  data?: Record<string, unknown>;
}

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
  streaming?: boolean;
  agentSteps?: AgentStep[];
}

const routeLabel: Record<string, string> = {
  general_chat: '通用对话',
  favorite_knowledge_query: '收藏夹问答',
  video_knowledge_query: '视频问答',
  import_request: '导入任务',
  retry_request: '重试任务',
  knowledge_query: '知识问答',
  plan_and_solve: '执行任务',
};

const routeTagClass: Record<string, string> = {
  general_chat: 'tag-gray',
  favorite_knowledge_query: 'tag-orange',
  video_knowledge_query: 'tag-blue',
  import_request: 'tag-green',
  retry_request: 'tag-yellow',
  knowledge_query: 'tag-blue',
  plan_and_solve: 'tag-green',
};

function genId() { return Math.random().toString(36).slice(2); }

const nodeLabel: Record<string, string> = {
  load_context: '加载上下文',
  router: '路由分析',
  general_chat: '通用对话',
  retrieve_knowledge: '知识检索',
  knowledge_qa: '知识问答',
  plan_and_solve: '制定计划',
  finalize_run: '完成',
};

const nodeIcon: Record<string, string> = {
  load_context: '⟳',
  router: '⊞',
  general_chat: '◎',
  retrieve_knowledge: '⬡',
  knowledge_qa: '◎',
  plan_and_solve: '⊙',
  finalize_run: '✓',
};

function AgentStatusBar({ steps, streaming }: { steps: AgentStep[]; streaming?: boolean }) {
  if (steps.length === 0 && !streaming) return null;
  return (
    <div className="agent-status-bar">
      {steps.map((step) => (
        <span key={step.node} className="agent-step done">
          <span className="agent-step-icon">{nodeIcon[step.node] ?? '●'}</span>
          <span className="agent-step-label">{nodeLabel[step.node] ?? step.node}</span>
        </span>
      ))}
      {streaming && (
        <span className="agent-step running">
          <span className="agent-step-spinner" />
          <span className="agent-step-label">思考中…</span>
        </span>
      )}
    </div>
  );
}

export default function ChatPage() {
  const navigate = useNavigate();
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [confirming, setConfirming] = useState<Record<string, boolean>>({});
  const bottomRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);

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

  const addMsg = useCallback((m: Message) => setMessages(prev => [...prev, m]), []);

  const appendToken = useCallback((token: string) => {
    setMessages(prev => {
      if (prev.length === 0) return prev;
      const last = prev[prev.length - 1];
      if (last.role === 'assistant' && last.streaming) {
        return [...prev.slice(0, -1), { ...last, content: last.content + token }];
      }
      return prev;
    });
  }, []);

  const updateAgentStep = useCallback((node: string, data: Record<string, unknown>) => {
    setMessages(prev => {
      if (prev.length === 0) return prev;
      const last = prev[prev.length - 1];
      if (last.role === 'assistant' && last.streaming) {
        const existingSteps = last.agentSteps ?? [];
        const existingIdx = existingSteps.findIndex(s => s.node === node);
        const newStep: AgentStep = { node, status: 'done', data };
        const newSteps = existingIdx >= 0
          ? existingSteps.map((s, i) => i === existingIdx ? newStep : s)
          : [...existingSteps, newStep];
        return [...prev.slice(0, -1), { ...last, agentSteps: newSteps }];
      }
      return prev;
    });
  }, []);

  const finalizeStreamingMsg = useCallback((done: StreamChunk & { type: 'done' }) => {
    setMessages(prev => {
      if (prev.length === 0) return prev;
      const last = prev[prev.length - 1];
      if (last.role === 'assistant' && last.streaming) {
        return [
          ...prev.slice(0, -1),
          {
            ...last,
            content: done.reply || last.content,
            route: done.route ?? last.route,
            runId: last.runId ?? done.run_id,
            status: done.status,
            executionPlan: done.execution_plan,
            requiresConfirmation: done.requires_confirmation,
            approvalStatus: done.approval_status ?? undefined,
            streaming: false,
          },
        ];
      }
      return prev;
    });
  }, []);

  const handleSend = async () => {
    const text = input.trim();
    if (!text || loading) return;
    setInput('');
    addMsg({ id: genId(), role: 'user', content: text });
    setLoading(true);

    const bubbleId = genId();
    addMsg({ id: bubbleId, role: 'assistant', content: '', streaming: true });

    abortRef.current = new AbortController();

    try {
      const done = await streamChat(
        { session_id: sessionId.current, user_id: userId, message: text },
        (chunk) => {
          if (chunk.type === 'token') {
            appendToken(chunk.content);
          } else if (chunk.type === 'node') {
            updateAgentStep(chunk.node, chunk.data);
          } else if (chunk.type === 'interrupt') {
            setMessages(prev => {
              const last = prev[prev.length - 1];
              if (last.role === 'assistant' && last.streaming) {
                return [
                  ...prev.slice(0, -1),
                  {
                    ...last,
                    executionPlan: (chunk.data as any).execution_plan,
                    requiresConfirmation: true,
                  },
                ];
              }
              return prev;
            });
          }
        },
        abortRef.current.signal,
      );

      if (done.run_id) {
        setMessages(prev => {
          const last = prev[prev.length - 1];
          if (last.role === 'assistant' && last.streaming) {
            return [...prev.slice(0, -1), { ...last, runId: done.run_id }];
          }
          return prev;
        });
      }

      finalizeStreamingMsg(done);
    } catch (e: any) {
      if (e?.name === 'AbortError') {
        setMessages(prev => {
          const last = prev[prev.length - 1];
          if (last.role === 'assistant' && last.streaming) {
            return [...prev.slice(0, -1), { ...last, content: last.content || '（已中断）', streaming: false }];
          }
          return prev;
        });
      } else {
        setMessages(prev => {
          const last = prev[prev.length - 1];
          if (last.role === 'assistant' && last.streaming) {
            return [...prev.slice(0, -1), { ...last, content: `❌ ${e?.message ?? '请求失败'}`, streaming: false }];
          }
          return prev;
        });
      }
    } finally {
      setLoading(false);
      abortRef.current = null;
    }
  };

  const handleConfirm = async (runId: string, approved: boolean) => {
    setConfirming(prev => ({ ...prev, [runId]: true }));

    setMessages(prev => prev.map(m =>
      m.runId === runId ? { ...m, requiresConfirmation: false } : m
    ));

    addMsg({ id: genId(), role: 'assistant', content: '', streaming: true });

    try {
      const done = await streamConfirm(
        runId,
        approved,
        (chunk) => {
          if (chunk.type === 'token') appendToken(chunk.content);
          else if (chunk.type === 'node') updateAgentStep(chunk.node, chunk.data);
        },
      );
      finalizeStreamingMsg(done);
    } catch (e: any) {
      setMessages(prev => {
        const last = prev[prev.length - 1];
        if (last.role === 'assistant' && last.streaming) {
          return [...prev.slice(0, -1), { ...last, content: `❌ ${e?.message ?? '确认失败'}`, streaming: false }];
        }
        return prev;
      });
    } finally {
      setConfirming(prev => ({ ...prev, [runId]: false }));
    }
  };

  const newSession = () => {
    abortRef.current?.abort();
    const id = genId();
    sessionId.current = id;
    localStorage.setItem('sessionId', id);
    setMessages([]);
    setLoading(false);
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
            <div className="chat-empty-badge">BIliBIl Agent</div>
            <div className="chat-empty-icon">◎</div>
            <h2 className="chat-empty-title">开始一次新的知识对话</h2>
            <p className="chat-empty-desc">你可以询问收藏夹内容、指定某个视频提问，或者直接发起导入任务。</p>
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
              {m.role === 'assistant' && (
                <AgentStatusBar steps={m.agentSteps ?? []} streaming={m.streaming && !m.content} />
              )}
              <div className="msg-content">
                {m.streaming && !m.content
                  ? <span className="stream-cursor" />
                  : <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.content}</ReactMarkdown>
                }
                {m.streaming && m.content && <span className="stream-cursor" />}
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
                      {confirming[m.runId] ? '处理中...' : '✓ 确认执行'}
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

        {loading && messages[messages.length - 1]?.streaming === false && (
          <div className="msg-row assistant">
            <div className="msg-bubble typing">
              <span /><span /><span />
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <div className="chat-input-bar">
        <div className="chat-input-wrap">
          <textarea
            className="chat-textarea"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); } }}
            placeholder="输入消息，Enter 发送，Shift+Enter 换行..."
            rows={2}
          />
          <div className="chat-input-hint">Enter 发送 · Shift+Enter 换行</div>
        </div>
        {loading ? (
          <button className="btn btn-ghost chat-stop-btn" onClick={() => abortRef.current?.abort()}>
            ■ 停止
          </button>
        ) : (
          <button className="btn btn-primary chat-send-btn" onClick={handleSend} disabled={!input.trim()}>
            发送
          </button>
        )}
      </div>
    </div>
  );
}
