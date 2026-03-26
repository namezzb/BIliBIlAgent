import { useEffect, useState, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { getRun } from '../api/agent';
import { apiBase } from '../api/client';
import './ImportProgressPage.css';

interface RunStep {
  step_key: string;
  step_name: string;
  status: string;
  input_summary?: string;
  output_summary?: string;
  updated_at: string;
}

const statusColor: Record<string, string> = {
  completed: 'tag-green',
  running: 'tag-orange',
  failed: 'tag-red',
};

export default function ImportProgressPage() {
  const { runId } = useParams<{ runId: string }>();
  const navigate = useNavigate();
  const [run, setRun] = useState<any>(null);
  const [events, setEvents] = useState<any[]>([]);
  const [done, setDone] = useState(false);
  const esRef = useRef<EventSource | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (!runId) return;

    // SSE
    const es = new EventSource(`${apiBase}/api/runs/${runId}/events?follow=true`);
    esRef.current = es;
    es.addEventListener('run_event', (e: MessageEvent) => {
      try {
        const ev = JSON.parse(e.data);
        setEvents(prev => [...prev, ev]);
        if (ev.type === 'run_completed' || ev.type === 'run_failed') {
          setDone(true);
          es.close();
        }
      } catch {}
    });
    es.onerror = () => { es.close(); setDone(true); };

    // Poll run detail
    const poll = () => getRun(runId).then(setRun).catch(() => {});
    poll();
    pollRef.current = setInterval(poll, 3000);

    return () => {
      es.close();
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [runId]);

  useEffect(() => {
    if (done && pollRef.current) {
      clearInterval(pollRef.current);
      if (runId) getRun(runId).then(setRun).catch(() => {});
    }
  }, [done]);

  const steps: RunStep[] = run?.steps ?? [];
  const status = run?.status ?? 'running';

  return (
    <div className="page import-page">
      <div className="page-header">
        <div>
          <h1 className="page-title">导入进度</h1>
          <p className="page-sub">{runId}</p>
        </div>
        <span className={`tag ${statusColor[status] ?? 'tag-gray'}`}>{status}</span>
      </div>

      <div className="import-layout">
        <div className="import-steps card">
          <h3 className="section-title">执行步骤</h3>
          {steps.length === 0 && !done && (
            <div className="center-loader"><div className="spinner" /><span>等待任务开始...</span></div>
          )}
          {steps.map((s, i) => (
            <div key={i} className="step-row">
              <span className={`tag ${statusColor[s.status] ?? 'tag-gray'}`}>{s.status}</span>
              <div className="step-info">
                <p className="step-name">{s.step_name}</p>
                {s.output_summary && <p className="step-summary">{s.output_summary}</p>}
              </div>
            </div>
          ))}
          {done && status === 'completed' && (
            <div className="done-banner success">✓ 导入完成</div>
          )}
          {done && status === 'failed' && (
            <div className="done-banner error">✗ 导入失败：{run?.reply}</div>
          )}
        </div>

        <div className="import-events card">
          <h3 className="section-title">实时事件</h3>
          <div className="events-list">
            {events.map((ev, i) => (
              <div key={i} className="event-row fade-in">
                <span className="tag tag-gray event-type">{ev.type}</span>
                <span className="event-time">{new Date(ev.timestamp).toLocaleTimeString()}</span>
                {ev.payload?.reply && <p className="event-reply">{ev.payload.reply}</p>}
              </div>
            ))}
            {events.length === 0 && <p className="event-empty">等待事件...</p>}
          </div>
        </div>
      </div>

      <div className="import-actions">
        <button className="btn btn-ghost" onClick={() => navigate('/')}>← 返回收藏夹</button>
        <button className="btn btn-primary" onClick={() => navigate('/chat')}>去对话 →</button>
      </div>
    </div>
  );
}
