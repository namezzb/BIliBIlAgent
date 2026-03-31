import { useEffect, useState, useCallback } from 'react';
import { useParams, useLocation, useNavigate } from 'react-router-dom';
import { listFolderVideos, submitImport } from '../api/bilibili';
import './VideosPage.css';

interface VideoItem {
  item_id: string;
  video_id?: string;
  bvid?: string;
  title: string;
  cover?: string;
  upper_name?: string;
  duration: number;
  fav_time?: number;
  selectable: boolean;
  unsupported_reason?: string;
}

function formatDuration(s: number) {
  const m = Math.floor(s / 60);
  const sec = s % 60;
  return `${m}:${sec.toString().padStart(2, '0')}`;
}

export default function VideosPage() {
  const { folderId } = useParams<{ folderId: string }>();
  const location = useLocation();
  const navigate = useNavigate();
  const folder = location.state?.folder;

  const [items, setItems] = useState<VideoItem[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [keyword, setKeyword] = useState('');
  const [order, setOrder] = useState('mtime');
  const [loading, setLoading] = useState(false);
  const [importing, setImporting] = useState(false);
  const [error, setError] = useState('');

  const fetchVideos = useCallback(async (pn: number, kw: string, ord: string) => {
    if (!folderId) return;
    setLoading(true);
    setError('');
    try {
      const d = await listFolderVideos(folderId, { pn, ps: 20, keyword: kw, order: ord });
      setItems(d.items ?? []);
      setTotalPages(d.total_pages ?? 1);
    } catch (e: any) {
      setError(e?.response?.data?.detail ?? '加载失败');
    } finally {
      setLoading(false);
    }
  }, [folderId]);

  useEffect(() => { fetchVideos(page, keyword, order); }, [page, order]);

  const handleSearch = () => { setPage(1); fetchVideos(1, keyword, order); };

  const toggleSelect = (id: string) => {
    setSelected((prev) => {
      const s = new Set(prev);
      if (s.has(id)) s.delete(id); else s.add(id);
      return s;
    });
  };

  const selectableItems = items.filter((i) => i.selectable && i.video_id);

  const toggleAll = () => {
    if (selected.size === selectableItems.length) {
      setSelected(new Set());
    } else {
      setSelected(new Set(selectableItems.map((i) => i.video_id!)));
    }
  };

  const handleImport = async () => {
    if (selected.size === 0 || !folderId) return;
    setImporting(true);
    try {
      const sessionId = localStorage.getItem('sessionId') ?? undefined;
      const userId = localStorage.getItem('userId') ?? undefined;
      const res = await submitImport({
        session_id: sessionId,
        user_id: userId,
        favorite_folder_id: folderId,
        selected_video_ids: Array.from(selected),
      });
      navigate(`/tasks`);
    } catch (e: any) {
      setError(e?.response?.data?.detail ?? '提交导入失败');
      setImporting(false);
    }
  };

  return (
    <div className="page videos-page">
      <div className="page-header">
        <div>
          <button className="btn btn-ghost" style={{marginBottom:'0.5rem'}} onClick={() => navigate('/')}>
            ← 返回收藏夹
          </button>
          <h1 className="page-title">{folder?.title ?? '视频列表'}</h1>
          <p className="page-sub">勾选视频后点击导入</p>
        </div>
      </div>

      <div className="videos-toolbar">
        <div className="toolbar-search">
          <input
            value={keyword}
            onChange={(e) => setKeyword(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
            placeholder="搜索视频标题..."
          />
          <button className="btn btn-ghost" onClick={handleSearch}>搜索</button>
        </div>
        <div className="toolbar-right">
          <select value={order} onChange={(e) => { setOrder(e.target.value); setPage(1); }}>
            <option value="mtime">收藏时间</option>
            <option value="view">播放量</option>
            <option value="pubtime">发布时间</option>
          </select>
          <button className="btn btn-ghost" onClick={toggleAll}>
            {selected.size === selectableItems.length && selectableItems.length > 0 ? '取消全选' : '全选'}
          </button>
        </div>
      </div>

      {error && <div className="error-banner">{error}</div>}

      {loading ? (
        <div className="center-loader"><div className="spinner" /><span>加载中...</span></div>
      ) : (
        <div className="videos-list">
          {items.map((item) => (
            <div
              key={item.item_id}
              className={`video-item${!item.selectable ? ' disabled' : ''}${selected.has(item.video_id ?? '') ? ' selected' : ''}`}
              onClick={() => item.selectable && item.video_id && toggleSelect(item.video_id)}
            >
              <div className="video-check">
                {item.selectable
                  ? <div className={`check-box${selected.has(item.video_id ?? '') ? ' checked' : ''}`}>{selected.has(item.video_id ?? '') ? '✓' : ''}</div>
                  : <div className="check-box disabled-box">—</div>
                }
              </div>
              <div className="video-cover">
                {item.cover
                  ? <img src={item.cover + '@200w'} alt={item.title} />
                  : <div className="video-cover-ph">▶</div>
                }
              </div>
              <div className="video-meta">
                <p className="video-title">{item.title}</p>
                <div className="video-tags">
                  {item.upper_name && <span className="tag tag-gray">{item.upper_name}</span>}
                  <span className="tag tag-gray">{formatDuration(item.duration)}</span>
                  {!item.selectable && <span className="tag tag-red">{item.unsupported_reason ?? '不支持'}</span>}
                  {item.bvid && <span className="tag tag-blue">{item.bvid}</span>}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      <div className="pagination">
        <button className="btn btn-ghost" disabled={page <= 1} onClick={() => setPage(p => p - 1)}>← 上一页</button>
        <span className="page-info">{page} / {totalPages}</span>
        <button className="btn btn-ghost" disabled={page >= totalPages} onClick={() => setPage(p => p + 1)}>下一页 →</button>
      </div>

      <div className="import-bar">
        <span className="import-count">已选 <strong>{selected.size}</strong> 个视频</span>
        <button
          className="btn btn-primary"
          disabled={selected.size === 0 || importing}
          onClick={handleImport}
        >
          {importing ? <><div className="spinner" style={{width:16,height:16}} /> 提交中...</> : `导入选中视频 (${selected.size})`}
        </button>
      </div>
    </div>
  );
}
