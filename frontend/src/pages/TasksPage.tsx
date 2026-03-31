import { useCallback, useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { listImportRuns } from '../api/agent';
import './TasksPage.css';

interface ImportRunItem {
  run_id: string;
  session_id: string;
  intent?: string;
  route?: string;
  status: string;
  latest_reply?: string;
  created_at: string;
  updated_at: string;
  total_items: number;
  indexed_count: number;
  needs_asr_count: number;
  duplicate_count: number;
  failed_count: number;
}

type TaskFilter = 'all' | 'running' | 'awaiting_confirmation' | 'completed' | 'failed';

const statusClass: Record<string, string> = {
  running: 'tag-orange',
  completed: 'tag-green',
  failed: 'tag-red',
  awaiting_confirmation: 'tag-blue',
  cancelled: 'tag-gray',
};

const statusLabel: Record<string, string> = {
  running: '进行中',
  completed: '已完成',
  failed: '失败',
  awaiting_confirmation: '待确认',
  cancelled: '已取消',
};

const filterOptions: { id: TaskFilter; label: string }[] = [
  { id: 'all', label: '全部' },
  { id: 'running', label: '进行中' },
  { id: 'awaiting_confirmation', label: '待确认' },
  { id: 'completed', label: '已完成' },
  { id: 'failed', label: '失败' },
];

function formatTime(value: string) {
  try {
    return new Date(value).toLocaleString();
  } catch {
    return value;
  }
}

function getTaskPriority(status: string) {
  if (status === 'running') return 0;
  if (status === 'awaiting_confirmation') return 1;
  if (status === 'failed') return 2;
  if (status === 'completed') return 3;
  return 4;
}

function summarizeTask(task: ImportRunItem) {
  const total = Number(task.total_items || 0);
  const indexed = Number(task.indexed_count || 0);
  const needsAsr = Number(task.needs_asr_count || 0);
  const duplicated = Number(task.duplicate_count || 0);
  const failed = Number(task.failed_count || 0);

  if (task.status === 'awaiting_confirmation') return '任务已生成执行计划，等待你确认后开始导入。';
  if (task.status === 'cancelled') return '该任务已取消，没有执行导入。';
  if (task.status === 'failed') return failed > 0 ? `任务执行失败，${failed} 个视频处理失败。` : '任务执行失败，建议查看详情排查原因。';
  if (task.status === 'running') return total > 0 ? `正在处理 ${total} 个视频，请稍候查看最新结果。` : '任务已提交，正在准备执行。';
  if (total === 0) return '任务已完成，但没有新增可展示的视频结果。';

  const parts: string[] = [];
  if (indexed > 0) parts.push(`${indexed} 个已入库`);
  if (needsAsr > 0) parts.push(`${needsAsr} 个进入语音识别`);
  if (duplicated > 0) parts.push(`${duplicated} 个已存在`);
  if (failed > 0) parts.push(`${failed} 个失败`);
  return parts.length > 0 ? `本次导入结果：${parts.join('，')}` : '任务已完成。';
}

export default function TasksPage() {
  const navigate = useNavigate();
  const [items, setItems] = useState<ImportRunItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState('');
  const [filter, setFilter] = useState<TaskFilter>('all');

  const loadItems = useCallback(async (silent = false) => {
    if (silent) setRefreshing(true);
    else setLoading(true);
    try {
      const data = await listImportRuns();
      setItems(data.items ?? []);
      setError('');
    } catch (e: any) {
      setError(e?.response?.data?.detail ?? '加载任务失败');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    loadItems();
    const timer = setInterval(() => loadItems(true), 5000);
    return () => clearInterval(timer);
  }, [loadItems]);

  const overview = useMemo(() => {
    const totalRuns = items.length;
    const runningRuns = items.filter((i) => i.status === 'running').length;
    const waitingRuns = items.filter((i) => i.status === 'awaiting_confirmation').length;
    const totalVideos = items.reduce((sum, i) => sum + Number(i.total_items || 0), 0);
    return { totalRuns, runningRuns, waitingRuns, totalVideos };
  }, [items]);

  const visibleItems = useMemo(() => {
    const filtered = filter === 'all' ? items : items.filter((item) => item.status === filter);
    return [...filtered].sort((a, b) => {
      const priorityDiff = getTaskPriority(a.status) - getTaskPriority(b.status);
      if (priorityDiff !== 0) return priorityDiff;
      return new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime();
    });
  }, [filter, items]);

  return (
    <div className="page tasks-page">
      <div className="page-header tasks-header">
        <div>
          <h1 className="page-title">任务</h1>
          <p className="page-sub">查看所有导入任务的执行状态与阶段结果</p>
        </div>
        <button className="btn btn-ghost" onClick={() => loadItems(true)}>
          {refreshing ? '刷新中...' : '刷新'}
        </button>
      </div>

      {!loading && !error && items.length > 0 && (
        <>
          <div className="task-overview-grid fade-in">
            <div className="card overview-card">
              <span className="overview-label">总任务数</span>
              <strong className="overview-value">{overview.totalRuns}</strong>
            </div>
            <div className="card overview-card overview-orange">
              <span className="overview-label">进行中</span>
              <strong className="overview-value">{overview.runningRuns}</strong>
            </div>
            <div className="card overview-card overview-blue">
              <span className="overview-label">待确认</span>
              <strong className="overview-value">{overview.waitingRuns}</strong>
            </div>
            <div className="card overview-card overview-green">
              <span className="overview-label">累计视频</span>
              <strong className="overview-value">{overview.totalVideos}</strong>
            </div>
          </div>

          <div className="task-filters fade-in">
            {filterOptions.map((option) => (
              <button
                key={option.id}
                className={`task-filter-chip ${filter === option.id ? 'active' : ''}`}
                onClick={() => setFilter(option.id)}
              >
                {option.label}
              </button>
            ))}
          </div>
        </>
      )}

      {loading && (
        <div className="center-loader"><div className="spinner" /><span>加载任务中...</span></div>
      )}

      {error && <div className="error-banner">{error}</div>}

      {!loading && !error && items.length === 0 && (
        <div className="card tasks-empty">还没有导入任务，先去收藏夹选择视频导入吧。</div>
      )}

      {!loading && !error && items.length > 0 && visibleItems.length === 0 && (
        <div className="card tasks-empty">当前筛选条件下没有任务。</div>
      )}

      {!loading && !error && visibleItems.length > 0 && (
        <div className="tasks-grid">
          {visibleItems.map((task, index) => {
            const settled = Number(task.indexed_count || 0) + Number(task.needs_asr_count || 0) + Number(task.duplicate_count || 0) + Number(task.failed_count || 0);
            const total = Number(task.total_items || 0);
            const pct = total > 0 ? Math.round((settled / total) * 100) : 0;
            return (
              <div
                key={task.run_id}
                className={`card task-card fade-in ${index === 0 ? 'task-card-highlight' : ''}`}
                onClick={() => navigate(`/import/${task.run_id}`)}
              >
                <div className="task-card-top">
                  <div>
                    <div className="task-title-row">
                      <p className="task-label">导入任务</p>
                      {index === 0 && <span className="task-pin">优先关注</span>}
                    </div>
                    <h3 className="task-title">任务 #{task.run_id.slice(0, 8)}</h3>
                    <p className="task-id">{task.run_id}</p>
                  </div>
                  <span className={`tag ${statusClass[task.status] ?? 'tag-gray'}`}>{statusLabel[task.status] ?? task.status}</span>
                </div>

                <p className="task-reply">{summarizeTask(task)}</p>

                <div className="task-progress">
                  <div className="task-progress-track">
                    <div className="task-progress-fill" style={{ width: `${pct}%` }} />
                  </div>
                  <span className="task-progress-text">{total > 0 ? `${settled} / ${total} 已处理` : '暂无视频结果'}</span>
                </div>

                <div className="task-stats">
                  <span className="task-stat success">已入库 {task.indexed_count || 0}</span>
                  <span className="task-stat info">待语音识别 {task.needs_asr_count || 0}</span>
                  <span className="task-stat mute">已存在 {task.duplicate_count || 0}</span>
                  <span className="task-stat danger">失败 {task.failed_count || 0}</span>
                </div>

                <div className="task-meta">
                  <span>最近更新 {formatTime(task.updated_at)}</span>
                  <span className="btn btn-ghost task-open-btn">查看详情 →</span>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
