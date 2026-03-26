import { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import QRCode from 'qrcode';
import { startQrLogin, pollQrLogin } from '../api/bilibili';
import './LoginPage.css';

type QrStatus = 'idle' | 'loading' | 'pending_scan' | 'scanned_waiting_confirm' | 'expired' | 'success' | 'error';

export default function LoginPage() {
  const navigate = useNavigate();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [status, setStatus] = useState<QrStatus>('idle');
  const [message, setMessage] = useState('');
  const [countdown, setCountdown] = useState(180);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const countRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const qrcodeKeyRef = useRef('');

  const clearTimers = () => {
    if (pollRef.current) clearInterval(pollRef.current);
    if (countRef.current) clearInterval(countRef.current);
  };

  const startLogin = async () => {
    clearTimers();
    setStatus('loading');
    setCountdown(180);
    try {
      const data = await startQrLogin();
      qrcodeKeyRef.current = data.qrcode_key;
      if (canvasRef.current) {
        await QRCode.toCanvas(canvasRef.current, data.qr_url, {
          width: 200,
          color: { dark: '#e8eaf2', light: '#13161e' },
        });
      }
      setStatus('pending_scan');
      setMessage('请用 B 站 App 扫码登录');

      countRef.current = setInterval(() => {
        setCountdown((c) => {
          if (c <= 1) { clearTimers(); setStatus('expired'); setMessage('二维码已过期，请刷新重试'); return 0; }
          return c - 1;
        });
      }, 1000);

      pollRef.current = setInterval(async () => {
        try {
          const res = await pollQrLogin(qrcodeKeyRef.current);
          if (res.status === 'scanned_waiting_confirm') {
            setStatus('scanned_waiting_confirm');
            setMessage('已扫码，请在手机上确认登录');
          } else if (res.status === 'expired') {
            clearTimers();
            setStatus('expired');
            setMessage('二维码已过期，请刷新重试');
          } else if (res.status === 'success') {
            clearTimers();
            setStatus('success');
            setMessage('登录成功！正在跳转...');
            localStorage.setItem('biliCookie', res.cookie);
            if (res.account) localStorage.setItem('biliAccount', JSON.stringify(res.account));
            setTimeout(() => navigate('/'), 800);
          }
        } catch {
          // transient errors are ok
        }
      }, 2000);
    } catch (e: any) {
      setStatus('error');
      setMessage(e?.response?.data?.detail ?? '获取二维码失败，请检查后端服务');
    }
  };

  useEffect(() => {
    if (localStorage.getItem('biliCookie')) { navigate('/'); return; }
    startLogin();
    return () => clearTimers();
  }, []);

  const statusLabel: Record<QrStatus, string> = {
    idle: '',
    loading: '生成二维码中...',
    pending_scan: '等待扫码',
    scanned_waiting_confirm: '请在手机上确认',
    expired: '已过期',
    success: '登录成功',
    error: '发生错误',
  };

  const statusClass: Record<QrStatus, string> = {
    idle: '', loading: 'tag-gray', pending_scan: 'tag-blue',
    scanned_waiting_confirm: 'tag-orange', expired: 'tag-red',
    success: 'tag-green', error: 'tag-red',
  };

  return (
    <div className="login-page">
      <div className="login-bg" />
      <div className="login-card fade-in">
        <div className="login-header">
          <div className="login-logo">B</div>
          <div>
            <h1 className="login-title">BIliBIl Agent</h1>
            <p className="login-sub">连接你的 B 站收藏夹，开始 AI 问答</p>
          </div>
        </div>

        <div className="qr-area">
          <canvas ref={canvasRef} className="qr-canvas" />
          {status === 'loading' && (
            <div className="qr-overlay"><div className="spinner" /></div>
          )}
          {status === 'expired' && (
            <div className="qr-overlay expired">
              <span>已过期</span>
            </div>
          )}
          {status === 'success' && (
            <div className="qr-overlay success">
              <span>✓</span>
            </div>
          )}
        </div>

        <div className="login-status">
          {status !== 'idle' && status !== 'loading' && (
            <span className={`tag ${statusClass[status]}`}>{statusLabel[status]}</span>
          )}
          <p className="login-message">{message}</p>
          {(status === 'pending_scan' || status === 'scanned_waiting_confirm') && (
            <div className="countdown">{countdown}s</div>
          )}
        </div>

        {(status === 'expired' || status === 'error') && (
          <button className="btn btn-primary" style={{ width: '100%', justifyContent: 'center' }} onClick={startLogin}>
            刷新二维码
          </button>
        )}

        <p className="login-hint">登录态仅保存在本地浏览器，不会上传至服务器</p>
      </div>
    </div>
  );
}
