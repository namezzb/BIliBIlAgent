import { useEffect, useState } from 'react';
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

const statusClass: Record<string, string> = {
  running: 'tag-orange',
  completed: 'tag-green',
  failed: 'tag-red',
  awaiting_confirmation: 'tag-blue',
};

function formatTime(value: string) {
  try {
    return new Date(value).toLocaleString();
  } catch {
    return value;
  }
}

export default function TasksPage() {
  const navigate = useNavigate();
  const [items, setItems] = useState<ImportRunItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    listImportRuns()
      .then((data) => {
        setItems(data.items ?? []);
        setLoading(false);
      })
      .catch((e: any) => {
        setError(e?.response?.data?.detail ?? '加载任务失败');
        setLoading(false);
      });
  }, []);

  return (
    <div className="page tasks-page">
      <div className="page-header">
        <div>
          <h1 className="page-title">任务</h1>
          <p className="page-sub">查看所有导入任务的执行状态与阶段结果</p>
        </div>
      </div>

      {loading && (
        <div className="center-loader"><div className="spinner" /><span>加载任务中...</span></div>
      )}

      {error && <div className="error-banner">{error}</div>}

      {!loading && !error && items.length === 0 && (
        <div className="card tasks-empty">还没有导入任务，先去收藏夹选择视频导入吧。</div>
      )}

      {!loading && !error && items.length > 0 && (
        <div className="tasks-grid">
          {items.map((task) => {
            const settled = Number(task.indexed_count || 0) + Number(task.needs_asr_count || 0) + Number(task.duplicate_count || 0) + Number(task.failed_count || 0);
            const total = Number(task.total_items || 0);
            const pct = total > 0 ? Math.round((settled / total) * 100) : 0;
            return (
              <div
                key={task.run_id}
                className="card task-card fade-in"
                onClick={() => navigate(`/import/${task.run_id}`)}
              >
                <div className="task-card-top">
                  <div>
                    <p className="task-label">导入任务</p>
                    <h3 className="task-title">{task.run_id}</h3>
                  </div>
                  <span className={`tag ${statusClass[task.status] ?? 'tag-gray'}`}>{task.status}</span>
                </div>

                <div className="task-progress">
                  <div className="task-progress-track">
                    <div className="task-progress-fill" style={{ width: `${pct}%` }} />
                  </div>
                  <span className="task-progress-text">{settled} / {total} 已落定</span>
                </div>

                <div className="task-stats">
                  <span className="task-stat success">已入库 {task.indexed_count || 0}</span>
                  <span className="task-stat info">待语音识别 {task.needs_asr_count || 0}</span>
                  <span className="task-stat mute">已存在 {task.duplicate_count || 0}</span>
                  <span className="task-stat danger">失败 {task.failed_count || 0}</span>
                </div>

                {task.latest_reply && <p className="task-reply">{task.latest_reply}</p>}

                <div className="task-meta">
                  <span>创建于 {formatTime(task.created_at)}</span>
                  <button className="btn btn-ghost task-open-btn">查看详情 →</button>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
