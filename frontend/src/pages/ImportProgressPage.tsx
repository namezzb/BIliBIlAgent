import { useEffect, useState, useRef, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { getRun, getImportItems } from '../api/agent';
import './ImportProgressPage.css';

interface RunStep {
  step_key: string;
  step_name: string;
  status: string;
  input_summary?: string;
  output_summary?: string;
  updated_at: string;
}

interface ImportItem {
  video_id: string;
  bvid?: string;
  title: string;
  status: 'indexed' | 'needs_asr' | 'failed' | 'skipped_duplicate' | string;
  needs_asr: boolean;
  failure_reason?: string;
  retryable: boolean;
  updated_at: string;
}

const ITEM_STATUS_META: Record<string, { label: string; cls: string; icon: string }> = {
  indexed:           { label: '已入库',   cls: 'item-indexed',    icon: '✓' },
  needs_asr:         { label: '待语音识别', cls: 'item-asr',       icon: '◎' },
  skipped_duplicate: { label: '已存在',   cls: 'item-dup',        icon: '⊘' },
  failed:            { label: '失败',     cls: 'item-failed',     icon: '✗' },
  running:           { label: '处理中',   cls: 'item-running',    icon: '⟳' },
};

const STEP_STATUS: Record<string, string> = {
  completed: 'tag-green',
  running:   'tag-orange',
  failed:    'tag-red',
};

const RUN_STATUS_META: Record<string, { label: string; cls: string }> = {
  running:   { label: '进行中', cls: 'run-running' },
  completed: { label: '已完成', cls: 'run-done' },
  failed:    { label: '失败',   cls: 'run-failed' },
};

function ItemCard({ item }: { item: ImportItem }) {
  const meta = ITEM_STATUS_META[item.status] ?? { label: item.status, cls: 'item-running', icon: '…' };
  return (
    <div className={`item-card fade-in ${meta.cls}`}>
      <div className="item-card-header">
        <span className={`item-icon ${meta.cls}`}>{meta.icon}</span>
        <div className="item-title-wrap">
          <p className="item-title">{item.title}</p>
          {item.bvid && <span className="item-bvid">{item.bvid}</span>}
        </div>
        <span className={`item-badge ${meta.cls}`}>{meta.label}</span>
      </div>
      {item.failure_reason && (
        <p className="item-reason">{item.failure_reason}</p>
      )}
    </div>
  );
}

function KanbanColumn({
  title, icon, items, cls,
}: { title: string; icon: string; items: ImportItem[]; cls: string }) {
  return (
    <div className={`kanban-col ${cls}`}>
      <div className="kanban-col-header">
        <span className="kanban-icon">{icon}</span>
        <span className="kanban-col-title">{title}</span>
        <span className="kanban-count">{items.length}</span>
      </div>
      <div className="kanban-items">
        {items.length === 0 ? (
          <div className="kanban-empty">暂无</div>
        ) : (
          items.map((item) => <ItemCard key={item.video_id} item={item} />)
        )}
      </div>
    </div>
  );
}

export default function ImportProgressPage() {
  const { runId } = useParams<{ runId: string }>();
  const navigate = useNavigate();

  const [run, setRun] = useState<any>(null);
  const [items, setItems] = useState<ImportItem[]>([]);
  const [done, setDone] = useState(false);

  const pollRunRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const pollItemsRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const fetchRun = useCallback(() => {
    if (!runId) return;
    getRun(runId).then((r) => {
      setRun(r);
      if (r.status === 'completed' || r.status === 'failed') {
        setDone(true);
      }
    }).catch(() => {});
  }, [runId]);

  const fetchItems = useCallback(() => {
    if (!runId) return;
    getImportItems(runId).then((r) => {
      setItems(r.items ?? []);
    }).catch(() => {});
  }, [runId]);

  useEffect(() => {
    if (!runId) return;
    fetchRun();
    fetchItems();
    pollRunRef.current   = setInterval(fetchRun,   2000);
    pollItemsRef.current = setInterval(fetchItems, 1500);
    return () => {
      if (pollRunRef.current)   clearInterval(pollRunRef.current);
      if (pollItemsRef.current) clearInterval(pollItemsRef.current);
    };
  }, [runId, fetchRun, fetchItems]);

  // Stop polling once done, do one final fetch
  useEffect(() => {
    if (!done) return;
    if (pollRunRef.current)   clearInterval(pollRunRef.current);
    if (pollItemsRef.current) clearInterval(pollItemsRef.current);
    fetchRun();
    fetchItems();
  }, [done]);

  const steps: RunStep[] = run?.steps ?? [];
  const runStatus = run?.status ?? 'running';
  const runMeta = RUN_STATUS_META[runStatus] ?? { label: runStatus, cls: 'run-running' };

  // Partition items into kanban columns
  const indexed    = items.filter((i) => i.status === 'indexed');
  const processing = items.filter((i) => !['indexed','needs_asr','failed','skipped_duplicate'].includes(i.status));
  const needsAsr   = items.filter((i) => i.status === 'needs_asr');
  const dupes      = items.filter((i) => i.status === 'skipped_duplicate');
  const failed     = items.filter((i) => i.status === 'failed');

  // Summary progress bar
  const total     = items.length;
  const settled   = indexed.length + needsAsr.length + dupes.length + failed.length;
  const pct       = total > 0 ? Math.round((settled / total) * 100) : (done ? 100 : 0);

  return (
    <div className="page import-page">
      {/* ── Header ── */}
      <div className="page-header">
        <div className="header-left">
          <button className="btn btn-ghost back-btn" onClick={() => navigate('/')}>← 返回</button>
          <div>
            <h1 className="page-title">导入任务看板</h1>
            <p className="page-sub mono">{runId}</p>
          </div>
        </div>
        <div className={`run-status-badge ${runMeta.cls}`}>
          {runStatus === 'running' && <span className="pulse-dot" />}
          {runMeta.label}
        </div>
      </div>

      {/* ── Progress bar ── */}
      <div className="progress-track">
        <div
          className={`progress-fill ${runStatus === 'failed' ? 'fill-failed' : runStatus === 'completed' ? 'fill-done' : 'fill-running'}`}
          style={{ width: `${pct}%` }}
        />
        <span className="progress-label">
          {total > 0 ? `${settled} / ${total} 视频处理完毕` : done ? '完成' : '准备中...'}
        </span>
      </div>

      {/* ── Two-column layout: Kanban + Steps ── */}
      <div className="import-body">

        {/* ── Kanban Board ── */}
        <div className="kanban-board">
          <h2 className="section-heading">视频状态看板</h2>
          <div className="kanban-grid">
            {processing.length > 0 && (
              <KanbanColumn title="处理中" icon="⟳" items={processing} cls="col-processing" />
            )}
            <KanbanColumn title="已入库" icon="✓" items={indexed}  cls="col-indexed" />
            <KanbanColumn title="待语音识别" icon="◎" items={needsAsr} cls="col-asr" />
            <KanbanColumn title="已存在" icon="⊘" items={dupes}   cls="col-dup" />
            <KanbanColumn title="失败" icon="✗" items={failed}  cls="col-failed" />
          </div>
          {items.length === 0 && !done && (
            <div className="board-waiting">
              <div className="spinner" />
              <span>等待视频任务开始...</span>
            </div>
          )}
        </div>

        {/* ── Steps sidebar ── */}
        <div className="steps-sidebar">
          <h2 className="section-heading">执行步骤</h2>
          <div className="steps-list card">
            {steps.length === 0 && !done && (
              <div className="center-loader"><div className="spinner" /><span>等待...</span></div>
            )}
            {steps.map((s, i) => (
              <div key={i} className="step-row">
                <span className={`tag ${STEP_STATUS[s.status] ?? 'tag-gray'}`}>{s.status}</span>
                <div className="step-info">
                  <p className="step-name">{s.step_name}</p>
                  {s.output_summary && <p className="step-summary">{s.output_summary}</p>}
                </div>
              </div>
            ))}
          </div>

          {/* Final banner */}
          {done && runStatus === 'completed' && (
            <div className="done-banner success">
              <span>✓</span>
              <div>
                <strong>导入完成</strong>
                {run?.reply && <p>{run.reply}</p>}
              </div>
            </div>
          )}
          {done && runStatus === 'failed' && (
            <div className="done-banner error">
              <span>✗</span>
              <div>
                <strong>导入失败</strong>
                {run?.reply && <p>{run.reply}</p>}
              </div>
            </div>
          )}

          <div className="sidebar-actions">
            <button className="btn btn-ghost" onClick={() => navigate('/')}>← 收藏夹</button>
            <button className="btn btn-primary" onClick={() => navigate('/chat')}>去对话 →</button>
          </div>
        </div>
      </div>
    </div>
  );
}
