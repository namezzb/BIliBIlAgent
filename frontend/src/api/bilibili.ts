import { api } from './client';

export const startQrLogin = () =>
  api.post('/api/bilibili/auth/qr/start').then((r) => r.data);

export const pollQrLogin = (qrcodeKey: string) =>
  api.get('/api/bilibili/auth/qr/poll', { params: { qrcode_key: qrcodeKey } }).then((r) => r.data);

export const listFavoriteFolders = () =>
  api.get('/api/bilibili/favorite-folders').then((r) => r.data);

export const listFolderVideos = (
  folderId: string,
  params: { pn?: number; ps?: number; keyword?: string; order?: string }
) =>
  api
    .get(`/api/bilibili/favorite-folders/${folderId}/videos`, { params })
    .then((r) => r.data);

export const submitImport = (payload: {
  session_id?: string;
  user_id?: string;
  favorite_folder_id: string;
  selected_video_ids: string[];
}) => api.post('/api/bilibili/imports', payload).then((r) => r.data);
