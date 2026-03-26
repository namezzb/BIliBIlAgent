import { useEffect, useState } from 'react';
import { getUserMemory, patchUserMemory, deleteUserMemory } from '../api/agent';
import './MemoryPage.css';

type MemoryGroup = 'preferences' | 'aliases' | 'default_scopes';

interface MemoryEntry {
  value: string;
  source_type: string;
  confirmed: boolean;
  created_at: string;
  updated_at: string;
}

const groupLabel: Record<MemoryGroup, string> = {
  preferences: '偏好设置',
  aliases: '别名映射',
  default_scopes: '默认范围',
};

const groupDesc: Record<MemoryGroup, string> = {
  preferences: '语言偏好、回答风格等',
  aliases: '自定义名称 → 收藏夹/视频 ID',
  default_scopes: '默认检索的收藏夹范围',
};

export default function MemoryPage() {
  const [userId, setUserId] = useState(localStorage.getItem('userId') ?? '');
  const [userIdInput, setUserIdInput] = useState(localStorage.getItem('userId') ?? '');
  const [profile, setProfile] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [adding, setAdding] = useState<{ group: MemoryGroup; key: string; value: string } | null>(null);
  const [saving, setSaving] = useState(false);

  const loadMemory = (uid: string) => {
    if (!uid) return;
    setLoading(true);
    setError('');
    getUserMemory(uid)
      .then(setProfile)
      .catch((e: any) => setError(e?.response?.data?.detail ?? '加载失败'))
      .finally(() => setLoading(false));
  };

  useEffect(() => { if (userId) loadMemory(userId); }, [userId]);

  const applyUserId = () => {
    const uid = userIdInput.trim();
    if (!uid) return;
    localStorage.setItem('userId', uid);
    setUserId(uid);
  };

  const handleDelete = async (group: MemoryGroup, key: string) => {
    if (!userId) return;
    try {
      const p = await deleteUserMemory(userId, group, key);
      setProfile(p);
    } catch (e: any) {
      setError(e?.response?.data?.detail ?? '删除失败');
    }
  };

  const handleAdd = async () => {
    if (!adding || !userId) return;
    setSaving(true);
    try {
      const p = await patchUserMemory(userId, { [adding.group]: { [adding.key]: adding.value } });
      setProfile(p);
      setAdding(null);
    } catch (e: any) {
      setError(e?.response?.data?.detail ?? '保存失败');
    } finally {
      setSaving(false);
    }
  };

  const groups: MemoryGroup[] = ['preferences', 'aliases', 'default_scopes'];

  return (
    <div className="page memory-page">
      <div className="page-header">
        <div>
          <h1 className="page-title">长期记忆</h1>
          <p className="page-sub">管理 Agent 为你保存的偏好与设置</p>
        </div>
      </div>

      <div className="card user-id-card">
        <label className="uid-label">用户 ID</label>
        <div className="uid-row">
          <input
            value={userIdInput}
            onChange={e => setUserIdInput(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && applyUserId()}
            placeholder="输入用户 ID（如 user_001）"
            style={{ flex: 1 }}
          />
          <button className="btn btn-primary" onClick={applyUserId}>加载记忆</button>
        </div>
        <p className="uid-hint">用户 ID 保存在本地，用于跨会话恢复长期记忆</p>
      </div>

      {error && <div className="error-banner">{error}</div>}
      {loading && <div className="center-loader"><div className="spinner" /><span>加载中...</span></div>}

      {profile && (
        <div className="memory-groups">
          {groups.map(group => {
            const entries = Object.entries<MemoryEntry>(profile[group] ?? {});
            return (
              <div key={group} className="card memory-group">
                <div className="group-header">
                  <div>
                    <h3 className="group-title">{groupLabel[group]}</h3>
                    <p className="group-desc">{groupDesc[group]}</p>
                  </div>
                  <button
                    className="btn btn-ghost"
                    onClick={() => setAdding({ group, key: '', value: '' })}
                  >+ 添加</button>
                </div>

                {entries.length === 0 && <p className="group-empty">暂无记录</p>}

                {entries.map(([key, entry]) => (
                  <div key={key} className="memory-row">
                    <div className="memory-kv">
                      <span className="mem-key">{key}</span>
                      <span className="mem-arrow">→</span>
                      <span className="mem-value">{entry.value}</span>
                    </div>
                    <div className="memory-meta">
                      <span className={`tag ${entry.confirmed ? 'tag-green' : 'tag-gray'}`}>
                        {entry.source_type}
                      </span>
                      <button
                        className="btn btn-danger mem-del-btn"
                        onClick={() => handleDelete(group, key)}
                      >删除</button>
                    </div>
                  </div>
                ))}

                {adding?.group === group && (
                  <div className="add-row">
                    <input
                      placeholder="键名（如：回答语言）"
                      value={adding.key}
                      onChange={e => setAdding(a => a ? { ...a, key: e.target.value } : a)}
                    />
                    <input
                      placeholder="值（如：中文）"
                      value={adding.value}
                      onChange={e => setAdding(a => a ? { ...a, value: e.target.value } : a)}
                    />
                    <button className="btn btn-primary" disabled={saving} onClick={handleAdd}>
                      {saving ? '保存中...' : '保存'}
                    </button>
                    <button className="btn btn-ghost" onClick={() => setAdding(null)}>取消</button>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
