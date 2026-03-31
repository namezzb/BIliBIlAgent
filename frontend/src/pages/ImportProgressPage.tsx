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
  indexed:           { label: '已入库', cls: 'item-indexed', icon: '✓' },
  needs_asr:         { label: '待语音识别', cls: 'item-asr', icon: '◎' },
  skipped_duplicate: { label: '已存在', cls: 'item-dup', icon: '⊘' },
  failed:            { label: '失败', cls: 'item-failed', icon: '✗' },
  running:           { label: '处理中', cls: 'item-running', icon: '⟳' },
};

const STEP_STATUS: Record<string, string> = {
  completed: 'tag-green',
  running: 'tag-orange',
  failed: 'tag-red',
};

const STEP_STATUS_LABEL: Record<string, string> = {
  completed: '已完成',
  running: '进行中',
  failed: '失败',
};

const RUN_STATUS_META: Record<string, { label: string; cls: string }> = {
  running: { label: '进行中', cls: 'run-running' },
  completed: { label: '已完成', cls: 'run-done' },
  failed: { label: '失败', cls: 'run-failed' },
};

const STEP_NAME_LABEL: Record<string, string> = {
  import_submitted: '提交导入任务',
  validate_selection: '校验已选视频',
  'bilibili_import.execute_import': '执行导入流程',
};

function humanizeStepName(step: RunStep) {
  if (STEP_NAME_LABEL[step.step_name]) return STEP_NAME_LABEL[step.step_name];
  if (step.step_name.startsWith('import_item:')) return '处理单个视频';
  return step.step_name.replaceAll('_', ' ');
}

function humanizeStepSummary(step: RunStep) {
  const summary = step.output_summary ?? '';
  if (!summary) return '';

  if (step.step_name === 'import_submitted') {
    const match = summary.match(/accepted\s+(\d+)\s+selected video/);
    return match ? `已提交导入，包含 ${match[1]} 个视频。` : '导入请求已提交。';
  }

  if (step.step_name === 'validate_selection') {
    const match = summary.match(/validated\s+(\d+)\s+selected video/);
    return match ? `已校验 ${match[1]} 个待导入视频。` : '已完成视频选择校验。';
  }

  if (step.step_name === 'bilibili_import.execute_import') {
    const match = summary.match(/indexed=(\d+), needs_asr=(\d+), skipped_duplicate=(\d+), failed=(\d+)/);
    if (match) {
      return `导入结束：${match[1]} 个已入库，${match[2]} 个待语音识别，${match[3]} 个已存在，${match[4]} 个失败。`;
    }
    return '导入流程已执行完成。';
  }

  if (step.step_name.startsWith('import_item:')) {
    if (summary.includes('prepared ASR fallback job')) return '未找到字幕，已加入语音识别队列。';
    if (summary.includes('indexed')) return '该视频已成功入库。';
  }

  return summary;
}

function summarizeRunReply(reply?: string) {
  if (!reply) return '';
  const match = reply.match(/indexed=(\d+), needs_asr=(\d+), skipped_duplicate=(\d+), failed=(\d+)/);
  if (match) {
    return `本次任务完成：${match[1]} 个已入库，${match[2]} 个待语音识别，${match[3]} 个已存在，${match[4]} 个失败。`;
  }
  if (reply.startsWith('Import accepted.')) return '导入任务已提交，系统正在抓取字幕、准备语音识别并写入知识库。';
  return reply;
}

function humanizeFailureReason(reason?: string) {
  if (!reason) return '';
  if (reason === 'subtitle_missing') return '未找到可用字幕，需走语音识别流程。';
  return reason;
}

function formatRelativeTime(value?: string) {
  if (!value) return '刚刚';
  const diff = Date.now() - new Date(value).getTime();
  if (Number.isNaN(diff) || diff < 60_000) return '刚刚';
  const minutes = Math.floor(diff / 60_000);
  if (minutes < 60) return `${minutes} 分钟前`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours} 小时前`;
  const days = Math.floor(hours / 24);
  return `${days} 天前`;
}

function getItemHint(item: ImportItem) {
  if (item.status === 'indexed') return '字幕与内容已整理完成，知识库可直接查询。';
  if (item.status === 'needs_asr') return '暂未拿到可用字幕，系统已转入语音识别流程。';
  if (item.status === 'skipped_duplicate') return '知识库中已有同视频内容，因此本次未重复导入。';
  if (item.status === 'failed') return humanizeFailureReason(item.failure_reason) || '处理失败，建议稍后重试或查看日志。';
  return '该视频仍在处理中，请稍候刷新查看。';
}

function getRunConclusion(runStatus: string, items: ImportItem[], reply?: string) {
  const indexed = items.filter((i) => i.status === 'indexed').length;
  const needsAsr = items.filter((i) => i.status === 'needs_asr').length;
  const dupes = items.filter((i) => i.status === 'skipped_duplicate').length;
  const failed = items.filter((i) => i.status === 'failed').length;

  if (runStatus === 'running') return '任务正在执行中，结果会随着处理进度持续更新。';
  if (runStatus === 'failed') return failed > 0 ? `任务失败，当前有 ${failed} 个视频处理失败，建议检查失败原因。` : '任务执行失败，建议查看步骤信息排查原因。';
  if (needsAsr > 0) return `本次导入已完成，其中 ${needsAsr} 个视频未命中字幕，已进入语音识别流程。`;
  if (indexed > 0 || dupes > 0) return summarizeRunReply(reply) || '任务已完成，视频结果已更新。';
  return '任务已完成，但没有新增可展示的视频结果。';
}

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
      <p className="item-hint">{getItemHint(item)}</p>
      {item.failure_reason && item.status === 'failed' && (
        <p className="item-reason">{humanizeFailureReason(item.failure_reason)}</p>
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
    pollRunRef.current = setInterval(fetchRun, 2000);
    pollItemsRef.current = setInterval(fetchItems, 1500);
    return () => {
      if (pollRunRef.current) clearInterval(pollRunRef.current);
      if (pollItemsRef.current) clearInterval(pollItemsRef.current);
    };
  }, [runId, fetchRun, fetchItems]);

  useEffect(() => {
    if (!done) return;
    if (pollRunRef.current) clearInterval(pollRunRef.current);
    if (pollItemsRef.current) clearInterval(pollItemsRef.current);
    fetchRun();
    fetchItems();
  }, [done, fetchRun, fetchItems]);

  const steps: RunStep[] = run?.steps ?? [];
  const runStatus = run?.status ?? 'running';
  const runMeta = RUN_STATUS_META[runStatus] ?? { label: runStatus, cls: 'run-running' };

  const indexed = items.filter((i) => i.status === 'indexed');
  const processing = items.filter((i) => !['indexed', 'needs_asr', 'failed', 'skipped_duplicate'].includes(i.status));
  const needsAsr = items.filter((i) => i.status === 'needs_asr');
  const dupes = items.filter((i) => i.status === 'skipped_duplicate');
  const failed = items.filter((i) => i.status === 'failed');

  const total = items.length;
  const settled = indexed.length + needsAsr.length + dupes.length + failed.length;
  const pct = total > 0 ? Math.round((settled / total) * 100) : (done ? 100 : 0);
  const updatedAt = run?.updated_at ? new Date(run.updated_at).toLocaleString() : '刚刚';
  const relativeUpdatedAt = formatRelativeTime(run?.updated_at);
  const conclusion = getRunConclusion(runStatus, items, run?.reply);

  return (
    <div className="page import-page">
      <div className="page-header">
        <div className="header-left">
          <button className="btn btn-ghost back-btn" onClick={() => navigate('/tasks')}>← 返回任务列表</button>
          <div>
            <h1 className="page-title">导入任务看板</h1>
            <p className="page-sub mono">任务 ID：{runId}</p>
          </div>
        </div>
        <div className={`run-status-badge ${runMeta.cls}`}>
          {runStatus === 'running' && <span className="pulse-dot" />}
          {runMeta.label}
        </div>
      </div>

      <div className="progress-track">
        <div
          className={`progress-fill ${runStatus === 'failed' ? 'fill-failed' : runStatus === 'completed' ? 'fill-done' : 'fill-running'}`}
          style={{ width: `${pct}%` }}
        />
        <span className="progress-label">
          {total > 0 ? `${settled} / ${total} 个视频已处理` : done ? '任务已结束' : '准备中...'}
        </span>
      </div>

      <div className="refresh-meta">
        <span className={`refresh-indicator ${done ? 'idle' : 'live'}`}>{done ? '已停止自动刷新' : '自动刷新中'}</span>
        <span className="refresh-time">上次更新：{updatedAt}（{relativeUpdatedAt}）</span>
      </div>

      <div className={`run-conclusion-banner ${runStatus === 'failed' ? 'error' : runStatus === 'running' ? 'running' : 'success'}`}>
        <span>{runStatus === 'failed' ? '✗' : runStatus === 'running' ? '⟳' : '✓'}</span>
        <div>
          <strong>{runStatus === 'failed' ? '任务结论：导入失败' : runStatus === 'running' ? '任务结论：处理中' : '任务结论：导入完成'}</strong>
          <p>{conclusion}</p>
        </div>
      </div>

      <div className="run-overview-grid fade-in">
        <div className="card run-overview-card">
          <span className="run-overview-label">任务状态</span>
          <strong className="run-overview-value">{runMeta.label}</strong>
        </div>
        <div className="card run-overview-card">
          <span className="run-overview-label">视频总数</span>
          <strong className="run-overview-value">{total}</strong>
        </div>
        <div className="card run-overview-card">
          <span className="run-overview-label">当前进度</span>
          <strong className="run-overview-value">{pct}%</strong>
        </div>
        <div className="card run-overview-card">
          <span className="run-overview-label">最近更新</span>
          <strong className="run-overview-value run-overview-time">{updatedAt}</strong>
        </div>
      </div>

      <div className="import-body">
        <div className="kanban-board">
          <h2 className="section-heading">视频状态看板</h2>
          <div className="kanban-grid">
            {processing.length > 0 && (
              <KanbanColumn title="处理中" icon="⟳" items={processing} cls="col-processing" />
            )}
            <KanbanColumn title="已入库" icon="✓" items={indexed} cls="col-indexed" />
            <KanbanColumn title="待语音识别" icon="◎" items={needsAsr} cls="col-asr" />
            <KanbanColumn title="已存在" icon="⊘" items={dupes} cls="col-dup" />
            <KanbanColumn title="失败" icon="✗" items={failed} cls="col-failed" />
          </div>
          {items.length === 0 && !done && (
            <div className="board-waiting">
              <div className="spinner" />
              <span>等待视频任务开始...</span>
            </div>
          )}
        </div>

        <div className="steps-sidebar">
          <h2 className="section-heading">执行步骤</h2>
          <div className="steps-list card">
            {steps.length === 0 && !done && (
              <div className="center-loader"><div className="spinner" /><span>等待...</span></div>
            )}
            {steps.map((s, i) => (
              <div key={i} className="step-row">
                <div className="step-rail">
                  <span className="step-dot" />
                  {i !== steps.length - 1 && <span className="step-line" />}
                </div>
                <span className={`tag ${STEP_STATUS[s.status] ?? 'tag-gray'}`}>{STEP_STATUS_LABEL[s.status] ?? s.status}</span>
                <div className="step-info">
                  <div className="step-head">
                    <p className="step-name">{humanizeStepName(s)}</p>
                    <span className="step-time">{formatRelativeTime(s.updated_at)}</span>
                  </div>
                  {humanizeStepSummary(s) && <p className="step-summary">{humanizeStepSummary(s)}</p>}
                </div>
              </div>
            ))}
          </div>

          {done && runStatus === 'completed' && (
            <div className="done-banner success">
              <span>✓</span>
              <div>
                <strong>导入完成</strong>
                {run?.reply && <p>{summarizeRunReply(run.reply)}</p>}
              </div>
            </div>
          )}
          {done && runStatus === 'failed' && (
            <div className="done-banner error">
              <span>✗</span>
              <div>
                <strong>导入失败</strong>
                {run?.reply && <p>{summarizeRunReply(run.reply)}</p>}
              </div>
            </div>
          )}

          <div className="sidebar-actions">
            <button className="btn btn-ghost" onClick={() => navigate('/tasks')}>← 返回任务</button>
            <button className="btn btn-primary" onClick={() => navigate('/chat')}>去对话 →</button>
          </div>
        </div>
      </div>
    </div>
  );
}
