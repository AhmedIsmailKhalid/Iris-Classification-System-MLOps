// LocalStorage utilities for prediction history
const HISTORY_KEY = 'iris_prediction_history';
const MAX_HISTORY_SIZE = 10;

export const savePrediction = (features, prediction) => {
  try {
    const history = getPredictionHistory();
    
    const newEntry = {
      id: Date.now(),
      timestamp: new Date().toISOString(),
      features,
      prediction: prediction.prediction,
      confidence: prediction.confidence,
      probabilities: prediction.probabilities,
      feature_contributions: prediction.feature_contributions || null,  // ← ADD THIS
      model_version: prediction.model_version || 'v1.0.0',  // ← ADD THIS
    };
    
    // Add to beginning, keep only MAX_HISTORY_SIZE items
    const updated = [newEntry, ...history].slice(0, MAX_HISTORY_SIZE);
    
    localStorage.setItem(HISTORY_KEY, JSON.stringify(updated));
    
    return updated;
  } catch (error) {
    console.error('Failed to save prediction:', error);
    return getPredictionHistory();
  }
};

export const getPredictionHistory = () => {
  try {
    const stored = localStorage.getItem(HISTORY_KEY);
    return stored ? JSON.parse(stored) : [];
  } catch (error) {
    console.error('Failed to load prediction history:', error);
    return [];
  }
};

export const clearPredictionHistory = () => {
  try {
    localStorage.removeItem(HISTORY_KEY);
    return true;
  } catch (error) {
    console.error('Failed to clear prediction history:', error);
    return false;
  }
};

export const exportHistoryAsJSON = () => {
  const history = getPredictionHistory();
  const dataStr = JSON.stringify(history, null, 2);
  const blob = new Blob([dataStr], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  
  const link = document.createElement('a');
  link.href = url;
  link.download = `iris-predictions-${Date.now()}.json`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
};