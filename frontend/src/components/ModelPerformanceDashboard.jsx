import { useState, useEffect } from 'react';
import { Activity, TrendingUp, Target, Database, ChevronDown, ChevronUp } from 'lucide-react';
import { getModelInfo } from '../services/api';
import LoadingSpinner from './LoadingSpinner';
import ErrorMessage from './ErrorMessage';

const ModelPerformanceDashboard = () => {
  const [modelInfo, setModelInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [isExpanded, setIsExpanded] = useState(false);

  useEffect(() => {
    loadModelInfo();
  }, []);

  const loadModelInfo = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await getModelInfo();
      setModelInfo(data);
    } catch (err) {
      console.error('Failed to load model info:', err);
      setError('Failed to load model information');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="card">
        <LoadingSpinner message="Loading model info..." />
      </div>
    );
  }

  if (error) {
    return (
      <div className="card">
        <ErrorMessage message={error} />
      </div>
    );
  }

  if (!modelInfo) return null;

  return (
    <div className="card space-y-4">
      {/* Header */}
      <div 
        className="flex items-center justify-between cursor-pointer"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center space-x-2">
          <Activity className="w-6 h-6 text-purple-600" />
          <h3 className="text-lg font-bold text-gray-800">Model Performance</h3>
        </div>
        <button className="text-gray-600 hover:text-purple-600 transition-colors">
          {isExpanded ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
        </button>
      </div>

      {/* Collapsed View - Key Metrics */}
      <div className="grid grid-cols-3 gap-4">
        <div className="bg-green-50 rounded-lg p-4 text-center">
          <Target className="w-6 h-6 text-green-600 mx-auto mb-2" />
          <div className="text-2xl font-bold text-green-700">
            {(modelInfo.metrics.test_accuracy * 100).toFixed(1)}%
          </div>
          <div className="text-xs text-gray-600 mt-1">Test Accuracy</div>
        </div>

        <div className="bg-blue-50 rounded-lg p-4 text-center">
          <TrendingUp className="w-6 h-6 text-blue-600 mx-auto mb-2" />
          <div className="text-2xl font-bold text-blue-700">
            {(modelInfo.metrics.test_f1_macro * 100).toFixed(1)}%
          </div>
          <div className="text-xs text-gray-600 mt-1">F1 Score</div>
        </div>

        <div className="bg-purple-50 rounded-lg p-4 text-center">
          <Database className="w-6 h-6 text-purple-600 mx-auto mb-2" />
          <div className="text-2xl font-bold text-purple-700">
            {modelInfo.dataset.num_classes}
          </div>
          <div className="text-xs text-gray-600 mt-1">Classes</div>
        </div>
      </div>

      {/* Expanded View - Detailed Info */}
      {isExpanded && (
        <div className="space-y-4 pt-4 border-t animate-fade-in">
          {/* Model Details */}
          <div className="bg-gray-50 rounded-lg p-4 space-y-2">
            <h4 className="font-semibold text-gray-800 mb-3">Model Details</h4>
            
            <div className="grid grid-cols-2 gap-3 text-sm">
              <div>
                <span className="text-gray-600">Model Type:</span>
                <span className="ml-2 font-medium text-gray-800 capitalize">
                  {modelInfo.model_type.replace('_', ' ')}
                </span>
              </div>
              
              <div>
                <span className="text-gray-600">Version:</span>
                <span className="ml-2 font-medium text-gray-800">
                  {modelInfo.model_version}
                </span>
              </div>
              
              <div className="col-span-2">
                <span className="text-gray-600">Model ID:</span>
                <span className="ml-2 font-mono text-xs text-gray-800">
                  {modelInfo.model_id}
                </span>
              </div>
            </div>
          </div>

          {/* Training Metrics */}
          <div className="bg-gray-50 rounded-lg p-4 space-y-3">
            <h4 className="font-semibold text-gray-800 mb-3">Training Metrics</h4>
            
            <div className="space-y-2">
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-gray-600">Train Accuracy</span>
                  <span className="font-semibold text-gray-800">
                    {(modelInfo.metrics.train_accuracy * 100).toFixed(2)}%
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-gradient-to-r from-green-400 to-green-600 h-2 rounded-full"
                    style={{ width: `${modelInfo.metrics.train_accuracy * 100}%` }}
                  />
                </div>
              </div>

              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-gray-600">Test Accuracy</span>
                  <span className="font-semibold text-gray-800">
                    {(modelInfo.metrics.test_accuracy * 100).toFixed(2)}%
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-gradient-to-r from-blue-400 to-blue-600 h-2 rounded-full"
                    style={{ width: `${modelInfo.metrics.test_accuracy * 100}%` }}
                  />
                </div>
              </div>

              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-gray-600">F1 Score (Macro)</span>
                  <span className="font-semibold text-gray-800">
                    {(modelInfo.metrics.test_f1_macro * 100).toFixed(2)}%
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-gradient-to-r from-purple-400 to-purple-600 h-2 rounded-full"
                    style={{ width: `${modelInfo.metrics.test_f1_macro * 100}%` }}
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Dataset Info */}
          <div className="bg-gray-50 rounded-lg p-4">
            <h4 className="font-semibold text-gray-800 mb-3">Dataset Information</h4>
            
            <div className="space-y-2 text-sm">
              <div>
                <span className="text-gray-600">Dataset:</span>
                <span className="ml-2 font-medium text-gray-800 capitalize">
                  {modelInfo.dataset.name}
                </span>
              </div>
              
              <div>
                <span className="text-gray-600">Features:</span>
                <div className="mt-2 flex flex-wrap gap-2">
                  {modelInfo.dataset.features.map((feature) => (
                    <span
                      key={feature}
                      className="px-2 py-1 bg-white rounded text-xs text-gray-700 border border-gray-200"
                    >
                      {feature}
                    </span>
                  ))}
                </div>
              </div>

              <div>
                <span className="text-gray-600">Target Classes:</span>
                <div className="mt-2 flex flex-wrap gap-2">
                  {modelInfo.dataset.target_classes.map((cls) => (
                    <span
                      key={cls}
                      className="px-2 py-1 bg-purple-100 rounded text-xs text-purple-700 font-medium capitalize"
                    >
                      {cls}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Timestamp */}
          <div className="text-xs text-gray-500 text-center pt-2 border-t">
            Last trained: {new Date(modelInfo.timestamp).toLocaleString()}
          </div>
        </div>
      )}
    </div>
  );
};

export default ModelPerformanceDashboard;