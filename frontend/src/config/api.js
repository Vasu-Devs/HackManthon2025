// FastAPI AI Backend Configuration
const FASTAPI_CONFIG = {
  BASE_URL: 'http://127.0.0.1:8000',
  TIMEOUT: 30000,
  POLLING_INTERVAL: 2000,
  MAX_FILE_SIZE: 10 * 1024 * 1024, // 10MB
};

export const FASTAPI_ENDPOINTS = {
  // Health
  HEALTH: `${FASTAPI_CONFIG.BASE_URL}/health`,
  
  // Chat endpoints
  CHAT: `${FASTAPI_CONFIG.BASE_URL}/chat`,
  CHAT_STREAM: `${FASTAPI_CONFIG.BASE_URL}/chat_stream`,
  ADMIN_CHAT: `${FASTAPI_CONFIG.BASE_URL}/admin_chat`,
  
  // Document management
  DOCUMENTS: `${FASTAPI_CONFIG.BASE_URL}/documents`,
  UPLOAD_PDF: `${FASTAPI_CONFIG.BASE_URL}/upload_pdf_async`,
  UPLOAD_STATUS: `${FASTAPI_CONFIG.BASE_URL}/upload_status`,
  APPROVE_DOC: `${FASTAPI_CONFIG.BASE_URL}/approve_doc`,
  DELETE_DOC: `${FASTAPI_CONFIG.BASE_URL}/delete_doc`,
  
  // Analytics
  RECENT_ACTIVITIES: `${FASTAPI_CONFIG.BASE_URL}/recent_activities`,
  LOGS: `${FASTAPI_CONFIG.BASE_URL}/logs`,
  TEST_RETRIEVAL: `${FASTAPI_CONFIG.BASE_URL}/test_retrieval`,
};

export { FASTAPI_CONFIG };
export default FASTAPI_CONFIG;
