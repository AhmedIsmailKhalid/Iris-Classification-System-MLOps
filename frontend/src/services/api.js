import axios from 'axios';

// Create axios instance with base configuration
const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000',
  timeout: parseInt(import.meta.env.VITE_API_TIMEOUT) || 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for logging
api.interceptors.request.use(
  (config) => {
    console.log('API Request:', config.method.toUpperCase(), config.url);
    return config;
  },
  (error) => {
    console.error('Request Error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    console.log('API Response:', response.status, response.config.url);
    return response;
  },
  (error) => {
    console.error('Response Error:', error.response?.status, error.message);
    return Promise.reject(error);
  }
);

// API Methods
export const healthCheck = async () => {
  const response = await api.get('/health');
  return response.data;
};

export const getModelInfo = async () => {
  const response = await api.get('/api/v1/model/info');
  return response.data;
};

export const predict = async (features) => {
  const response = await api.post('/api/v1/predict', {
    features: features,
  });
  return response.data;
};

export const predictBatch = async (samples) => {
  const response = await api.post('/api/v1/predict/batch', {
    samples: samples,
  });
  return response.data;
};

export const compareFlowers = async (flower1Features, flower2Features) => {
  const response = await api.post('/api/v1/compare', {
    flower1: { features: flower1Features },
    flower2: { features: flower2Features },
  });
  return response.data;
};

// Monitoring endpoints
export const generateData = async (params) => {
  const response = await api.post('/api/v1/monitoring/generate-data', params);
  return response.data;
};

export const getDataStats = async () => {
  const response = await api.get('/api/v1/monitoring/data-stats');
  return response.data;
};

export const checkDrift = async () => {
  const response = await api.post('/api/v1/monitoring/check-drift');
  return response.data;
};

export default api;


// Retraining endpoints
export const triggerRetraining = async (force = false) => {
  const response = await api.post(`/api/v1/monitoring/trigger-retraining?force=${force}`);
  return response.data;
};

export const getWorkflowStatus = async () => {
  const response = await api.get('/api/v1/monitoring/workflow-status');
  return response.data;
};

// Model Registry
export const getModelRegistry = async () => {
  const response = await api.get('/api/v1/monitoring/model-registry');
  return response.data;
};