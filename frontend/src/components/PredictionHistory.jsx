import { History, Trash2, Download } from 'lucide-react';
import { useState, useEffect } from 'react';
import { 
  getPredictionHistory, 
  clearPredictionHistory, 
  exportHistoryAsJSON 
} from '../utils/storage';

const PredictionHistory = ({ onReload }) => {
  const [history, setHistory] = useState([]);
  const [isExpanded, setIsExpanded] = useState(false);

  useEffect(() => {
    loadHistory();
  }, []);

  const loadHistory = () => {
    const stored = getPredictionHistory();
    setHistory(stored);
  };

  const handleClear = () => {
    if (window.confirm('Clear all prediction history?')) {
      clearPredictionHistory();
      setHistory([]);
    }
  };

  const handleExport = () => {
    exportHistoryAsJSON();
  };

  const handleReload = (entry) => {
    if (onReload) {
      onReload(entry);  // Pass the entire entry object
    }
  };

  const formatTimestamp = (timestamp) => {
    const date = new Date(timestamp);
    return date.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const getSpeciesEmoji = (species) => {
    const emojis = {
      setosa: '🌸',
      versicolor: '🌺',
      virginica: '🌷',
    };
    return emojis[species.toLowerCase()] || '🌼';
  };

  if (history.length === 0) {
    return (
      <div className="card bg-gray-50 text-center text-gray-500">
        <History className="w-12 h-12 mx-auto mb-3 opacity-50" />
        <p>No predictions yet</p>
        <p className="text-sm mt-1">Your prediction history will appear here</p>
      </div>
    );
  }

  return (
    <div className="card space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <History className="w-6 h-6 text-purple-600" />
          <h3 className="text-lg font-bold text-gray-800">
            Prediction History
            <span className="text-sm font-normal text-gray-500 ml-2">
              ({history.length})
            </span>
          </h3>
        </div>
        
        <div className="flex items-center space-x-2">
          <button
            onClick={handleExport}
            className="p-2 text-gray-600 hover:text-purple-600 transition-colors"
            title="Export as JSON"
          >
            <Download className="w-5 h-5" />
          </button>
          <button
            onClick={handleClear}
            className="p-2 text-gray-600 hover:text-red-600 transition-colors"
            title="Clear history"
          >
            <Trash2 className="w-5 h-5" />
          </button>
        </div>
      </div>

      {/* History List */}
      <div className="space-y-2 max-h-96 overflow-y-auto">
        {history.slice(0, isExpanded ? history.length : 5).map((entry) => (
          <div
            key={entry.id}
            onClick={() => handleReload(entry)}
            className="bg-gray-50 hover:bg-purple-50 rounded-lg p-3 cursor-pointer transition-colors border border-transparent hover:border-purple-200"
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3 flex-1">
                <div className="text-2xl">{getSpeciesEmoji(entry.prediction)}</div>
                
                <div className="flex-1 min-w-0">
                  <div className="flex items-center space-x-2">
                    <span className="font-semibold text-gray-800 capitalize">
                      {entry.prediction}
                    </span>
                    <span className="text-sm text-gray-500">
                      {(entry.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                  
                  <div className="text-xs text-gray-500 mt-1 truncate">
                    SL: {entry.features['sepal length (cm)']}, 
                    SW: {entry.features['sepal width (cm)']}, 
                    PL: {entry.features['petal length (cm)']}, 
                    PW: {entry.features['petal width (cm)']}
                  </div>
                </div>
              </div>
              
              <span className="text-xs text-gray-400 ml-3">
                {formatTimestamp(entry.timestamp)}
              </span>
            </div>
          </div>
        ))}
      </div>

      {/* Show More/Less Button */}
      {history.length > 5 && (
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="w-full text-sm text-purple-600 hover:text-purple-700 font-medium py-2"
        >
          {isExpanded ? 'Show Less' : `Show ${history.length - 5} More`}
        </button>
      )}
    </div>
  );
};

export default PredictionHistory;