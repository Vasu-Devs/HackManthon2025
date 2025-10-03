import { FASTAPI_ENDPOINTS, FASTAPI_CONFIG } from '../config/api.js';

// Utility to get auth token (for FastAPI requests that need user context)
const getAuthHeaders = () => {
  const token = localStorage.getItem('authToken');
  return {
    'Content-Type': 'application/json',
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  };
};

// Handle API responses
const handleResponse = async (response) => {
  if (!response.ok) {
    const error = await response.text();
    throw new Error(`FastAPI Error: ${response.status} - ${error}`);
  }
  return response.json();
};

// FastAPI Service for AI/RAG functionality
export const fastAPIService = {
  // Health check
  async checkHealth() {
    try {
      const response = await fetch(FASTAPI_ENDPOINTS.HEALTH);
      return await response.json();
    } catch (error) {
      return { status: 'error', message: error.message };
    }
  },

  // Chat with streaming response
  async chatStream(message, department = 'General', k = 5) {
    const userId = localStorage.getItem('userRegNo') || 'default';
    
    const response = await fetch(FASTAPI_ENDPOINTS.CHAT_STREAM, {
      method: 'POST',
      headers: getAuthHeaders(),
      body: JSON.stringify({
        message,
        department,
        k,
        user_id: userId,
      }),
    });

    if (!response.ok) {
      throw new Error(`Chat stream failed: ${response.status}`);
    }

    return response.body.getReader();
  },

  // Regular chat (non-streaming)
  async chat(message, department = 'General', k = 5) {
    const userId = localStorage.getItem('userRegNo') || 'default';
    
    const response = await fetch(FASTAPI_ENDPOINTS.CHAT, {
      method: 'POST',
      headers: getAuthHeaders(),
      body: JSON.stringify({
        message,
        department,
        k,
        user_id: userId,
      }),
    });
    
    return handleResponse(response);
  },

  // Admin chat
  async adminChat(message, department = 'General', k = 5) {
    const response = await fetch(FASTAPI_ENDPOINTS.ADMIN_CHAT, {
      method: 'POST',
      headers: getAuthHeaders(),
      body: JSON.stringify({
        message,
        department,
        k,
      }),
    });
    
    return handleResponse(response);
  },

  // Upload PDF
  async uploadPDF(file) {
    if (!file || !file.name.toLowerCase().endsWith('.pdf')) {
      throw new Error('Only PDF files are allowed');
    }
    
    if (file.size > FASTAPI_CONFIG.MAX_FILE_SIZE) {
      throw new Error('File size exceeds 10MB limit');
    }

    const formData = new FormData();
    formData.append('file', file);

    const token = localStorage.getItem('authToken');
    const response = await fetch(FASTAPI_ENDPOINTS.UPLOAD_PDF, {
      method: 'POST',
      headers: {
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: formData,
    });

    return handleResponse(response);
  },

  // Check upload status
  async getUploadStatus(uploadId) {
    const response = await fetch(`${FASTAPI_ENDPOINTS.UPLOAD_STATUS}/${uploadId}`, {
      headers: getAuthHeaders(),
    });
    return handleResponse(response);
  },

  // Get all documents
  async getDocuments() {
    const response = await fetch(FASTAPI_ENDPOINTS.DOCUMENTS, {
      headers: getAuthHeaders(),
    });
    return handleResponse(response);
  },

  // Approve document (admin only)
  async approveDocument(filename) {
    const response = await fetch(`${FASTAPI_ENDPOINTS.APPROVE_DOC}/${filename}`, {
      method: 'POST',
      headers: getAuthHeaders(),
    });
    return handleResponse(response);
  },

  // Delete document (admin only)
  async deleteDocument(filename) {
    const response = await fetch(`${FASTAPI_ENDPOINTS.DELETE_DOC}/${filename}`, {
      method: 'DELETE',
      headers: getAuthHeaders(),
    });
    return handleResponse(response);
  },

  // ✅ Download document (updated to handle PDFs properly)
  async downloadDocument(filename) {
    const response = await fetch(`${FASTAPI_CONFIG.BASE_URL}/get_file/${filename}`, {
      headers: getAuthHeaders(),
    });

    if (!response.ok) {
      throw new Error(`Download failed: ${response.status}`);
    }

    const blob = await response.blob();
    const url = window.URL.createObjectURL(new Blob([blob], { type: "application/pdf" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.URL.revokeObjectURL(url);
  },

  // Get recent activities
  async getRecentActivities() {
    const response = await fetch(FASTAPI_ENDPOINTS.RECENT_ACTIVITIES, {
      headers: getAuthHeaders(),
    });
    return handleResponse(response);
  },

  // Get system logs
  async getLogs() {
    const response = await fetch(FASTAPI_ENDPOINTS.LOGS, {
      headers: getAuthHeaders(),
    });
    return handleResponse(response);
  },

  // Test document retrieval
  async testRetrieval(query) {
    const response = await fetch(FASTAPI_ENDPOINTS.TEST_RETRIEVAL, {
      method: 'POST',
      headers: getAuthHeaders(),
      body: JSON.stringify({ message: query }),
    });
    return handleResponse(response);
  },
};

// Stream processing utility
export const streamUtils = {
  async processStream(reader, onStatus, onDoc, onToken, onDone, onError) {
    const decoder = new TextDecoder();
    
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const lines = decoder.decode(value, { stream: true }).split('\n');
        for (let line of lines) {
          if (!line.trim()) continue;
          
          try {
            const event = JSON.parse(line);
            
            switch (event.type) {
              case 'status':
                onStatus && onStatus(event.message);
                break;
              case 'doc':
                onDoc && onDoc(event);
                break;
              case 'token':
                onToken && onToken(event.text);
                break;
              case 'done':
                onDone && onDone();
                return;
              default:
                console.warn('Unknown stream event:', event.type);
            }
          } catch (parseError) {
            console.warn('Failed to parse stream event:', parseError);
          }
        }
      }
    } catch (error) {
      onError && onError(error);
    }
  },
};

// Upload utility with progress tracking
export const uploadUtils = {
  async uploadWithProgress(file, onProgress = () => {}) {
    const uploadResponse = await fastAPIService.uploadPDF(file);
    const { upload_id } = uploadResponse;

    return new Promise((resolve, reject) => {
      const pollInterval = setInterval(async () => {
        try {
          const status = await fastAPIService.getUploadStatus(upload_id);
          onProgress(status);

          if (status.status === 'completed') {
            clearInterval(pollInterval);
            resolve(status);
          } else if (status.status === 'error') {
            clearInterval(pollInterval);
            reject(new Error(status.error || 'Upload failed'));
          }
        } catch (error) {
          clearInterval(pollInterval);
          reject(error);
        }
      }, FASTAPI_CONFIG.POLLING_INTERVAL);

      // Timeout after 5 minutes
      setTimeout(() => {
        clearInterval(pollInterval);
        reject(new Error('Upload timeout'));
      }, 300000);
    });
  },
};

export default fastAPIService;
