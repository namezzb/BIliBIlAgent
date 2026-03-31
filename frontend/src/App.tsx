import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import LoginPage from './pages/LoginPage';
import FoldersPage from './pages/FoldersPage';
import VideosPage from './pages/VideosPage';
import ImportProgressPage from './pages/ImportProgressPage';
import TasksPage from './pages/TasksPage';
import ChatPage from './pages/ChatPage';
import MemoryPage from './pages/MemoryPage';
import Layout from './components/Layout';

function RequireAuth({ children }: { children: React.ReactNode }) {
  const cookie = localStorage.getItem('biliCookie');
  if (!cookie) return <Navigate to="/login" replace />;
  return <>{children}</>;
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route element={<Layout />}>
          <Route path="/" element={<RequireAuth><FoldersPage /></RequireAuth>} />
          <Route path="/folders/:folderId/videos" element={<RequireAuth><VideosPage /></RequireAuth>} />
          <Route path="/tasks" element={<RequireAuth><TasksPage /></RequireAuth>} />
          <Route path="/import/:runId" element={<RequireAuth><ImportProgressPage /></RequireAuth>} />
          <Route path="/chat" element={<ChatPage />} />
          <Route path="/memory" element={<MemoryPage />} />
        </Route>
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}
