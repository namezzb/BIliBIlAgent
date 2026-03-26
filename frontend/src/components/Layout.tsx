import { Outlet, NavLink, useNavigate } from 'react-router-dom';
import './Layout.css';

export default function Layout() {
  const navigate = useNavigate();
  const account = (() => {
    try { return JSON.parse(localStorage.getItem('biliAccount') ?? '{}'); } catch { return {}; }
  })();

  const logout = () => {
    localStorage.removeItem('biliCookie');
    localStorage.removeItem('biliAccount');
    navigate('/login');
  };

  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="sidebar-logo">
          <span className="logo-mark">B</span>
          <span className="logo-text">BIliBIl<br/><small>Agent</small></span>
        </div>
        <nav className="sidebar-nav">
          <NavLink to="/" end className={({isActive}) => isActive ? 'nav-item active' : 'nav-item'}>
            <span className="nav-icon">⊞</span> 收藏夹
          </NavLink>
          <NavLink to="/chat" className={({isActive}) => isActive ? 'nav-item active' : 'nav-item'}>
            <span className="nav-icon">◎</span> 对话
          </NavLink>
          <NavLink to="/memory" className={({isActive}) => isActive ? 'nav-item active' : 'nav-item'}>
            <span className="nav-icon">⬡</span> 记忆
          </NavLink>
        </nav>
        <div className="sidebar-footer">
          {account.uname && <div className="account-info"><span className="acct-dot">●</span> {account.uname}</div>}
          <button className="btn btn-ghost" style={{width:'100%', justifyContent:'center'}} onClick={logout}>退出登录</button>
        </div>
      </aside>
      <main className="main-content">
        <Outlet />
      </main>
    </div>
  );
}
