import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { listFavoriteFolders } from '../api/bilibili';
import './FoldersPage.css';

interface Folder {
  favorite_folder_id: string;
  title: string;
  intro?: string;
  cover?: string;
  media_count: number;
}

export default function FoldersPage() {
  const navigate = useNavigate();
  const [folders, setFolders] = useState<Folder[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const account = (() => { try { return JSON.parse(localStorage.getItem('biliAccount') ?? '{}'); } catch { return {}; } })();

  useEffect(() => {
    listFavoriteFolders()
      .then((d) => { setFolders(d.folders ?? []); setLoading(false); })
      .catch((e) => { setError(e?.response?.data?.detail ?? '获取收藏夹失败'); setLoading(false); });
  }, []);

  return (
    <div className="page folders-page">
      <div className="page-header">
        <div>
          <h1 className="page-title">收藏夹</h1>
          <p className="page-sub">选择一个收藏夹导入视频知识</p>
        </div>
        {account.uname && <span className="tag tag-green">● {account.uname}</span>}
      </div>

      {loading && (
        <div className="center-loader"><div className="spinner" /><span>加载中...</span></div>
      )}

      {error && (
        <div className="error-banner">{error}</div>
      )}

      {!loading && !error && folders.length === 0 && (
        <div className="empty-state">暂无收藏夹</div>
      )}

      <div className="folders-grid">
        {folders.map((f) => (
          <div
            key={f.favorite_folder_id}
            className="folder-card fade-in"
            onClick={() => navigate(`/folders/${f.favorite_folder_id}/videos`, { state: { folder: f } })}
          >
            <div className="folder-cover">
              {f.cover
                ? <img src={f.cover + '@400w'} alt={f.title} />
                : <div className="folder-cover-placeholder">📁</div>
              }
              <span className="folder-count">{f.media_count}</span>
            </div>
            <div className="folder-info">
              <h3 className="folder-title">{f.title}</h3>
              {f.intro && <p className="folder-intro">{f.intro}</p>}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
