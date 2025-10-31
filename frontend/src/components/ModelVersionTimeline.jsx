import { useState, useEffect } from 'react';
import { History, TrendingUp, TrendingDown, Minus, Clock, CheckCircle, RefreshCw } from 'lucide-react';
import { getModelRegistry } from '../services/api';

const ModelVersionTimeline = () => {
  const [registry, setRegistry] = useState(null);
  const [loading, setLoading] = useState(true);
  const [selectedModel, setSelectedModel] = useState(null);

  useEffect(() => {
    loadRegistry();
  }, []);

  const loadRegistry = async () => {
    try {
      setLoading(true);
      const data = await getModelRegistry();
      setRegistry(data);
    } catch (error) {
      console.error('Failed to load model registry:', error);
    } finally {
      setLoading(false);
    }
  };

  const formatDate = (timestamp) => {
    const date = new Date(timestamp);
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const getAccuracyTrend = (currentModel, previousModel) => {
    if (!previousModel) return null;
    
    const current = currentModel.metrics.test_accuracy;
    const previous = previousModel.metrics.test_accuracy;
    const diff = current - previous;
    
    if (Math.abs(diff) < 0.001) {
      return { icon: Minus, color: 'text-gray-500', text: 'No change' };
    } else if (diff > 0) {
      return { 
        icon: TrendingUp, 
        color: 'text-green-600', 
        text: `+${(diff * 100).toFixed(2)}%` 
      };
    } else {
      return { 
        icon: TrendingDown, 
        color: 'text-red-600', 
        text: `${(diff * 100).toFixed(2)}%` 
      };
    }
  };

  const getModelColor = (modelId, activeModelId) => {
    if (modelId === activeModelId) {
      return 'border-green-500 bg-green-50';
    }
    return 'border-gray-300 bg-white hover:border-purple-300';
  };

  if (loading) {
    return (
      <div className="card">
        <div className="flex items-center justify-center py-8">
          <RefreshCw className="w-6 h-6 text-purple-600 animate-spin" />
          <span className="ml-2 text-gray-600">Loading model history...</span>
        </div>
      </div>
    );
  }

  if (!registry || !registry.success || registry.models.length === 0) {
    return (
      <div className="card">
        <div className="flex items-center space-x-2 mb-4">
          <History className="w-6 h-6 text-purple-600" />
          <h3 className="text-lg font-bold text-gray-800">Model Version Timeline</h3>
        </div>
        <div className="text-center py-8 text-gray-500">
          <History className="w-12 h-12 mx-auto mb-2 text-gray-300" />
          <p>No model versions yet</p>
          <p className="text-xs mt-1">Train a model to see version history</p>
        </div>
      </div>
    );
  }

  return (
    <div className="card space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <History className="w-6 h-6 text-purple-600" />
          <h3 className="text-lg font-bold text-gray-800">Model Version Timeline</h3>
        </div>
        <button
          onClick={loadRegistry}
          className="p-2 text-gray-600 hover:text-purple-600 transition-colors"
          title="Refresh"
        >
          <RefreshCw className="w-5 h-5" />
        </button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-3 gap-3">
        <div className="bg-purple-50 rounded-lg p-3 text-center">
          <div className="text-2xl font-bold text-purple-700">
            {registry.total_models}
          </div>
          <div className="text-xs text-gray-600 mt-1">Total Versions</div>
        </div>
        
        <div className="bg-green-50 rounded-lg p-3 text-center">
          <div className="text-2xl font-bold text-green-700">
            {(registry.models[0]?.metrics.test_accuracy * 100).toFixed(1)}%
          </div>
          <div className="text-xs text-gray-600 mt-1">Current Accuracy</div>
        </div>
        
        <div className="bg-blue-50 rounded-lg p-3 text-center">
          <div className="text-2xl font-bold text-blue-700">
            {registry.metadata?.num_classes || 3}
          </div>
          <div className="text-xs text-gray-600 mt-1">Classes</div>
        </div>
      </div>

      {/* Timeline */}
      <div className="relative space-y-3 max-h-96 overflow-y-auto">
        {/* Timeline line */}
        <div className="absolute left-4 top-0 bottom-0 w-0.5 bg-gradient-to-b from-purple-300 to-pink-300"></div>

        {registry.models.map((model, index) => {
          const isActive = model.model_id === registry.active_model;
          const trend = getAccuracyTrend(model, registry.models[index + 1]);
          const TrendIcon = trend?.icon;

          return (
            <div
              key={model.model_id}
              className={`relative pl-12 cursor-pointer transition-all ${
                selectedModel === model.model_id ? 'scale-105' : ''
              }`}
              onClick={() => setSelectedModel(
                selectedModel === model.model_id ? null : model.model_id
              )}
            >
              {/* Timeline dot */}
              <div className={`absolute left-2.5 top-3 w-3 h-3 rounded-full ${
                isActive ? 'bg-green-500 ring-4 ring-green-100' : 'bg-purple-400'
              }`}></div>

              {/* Model card */}
              <div className={`border-2 rounded-lg p-3 ${getModelColor(model.model_id, registry.active_model)}`}>
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-2 mb-1">
                      <span className="font-semibold text-gray-800 text-sm">
                        {model.model_id}
                      </span>
                      {isActive && (
                        <span className="flex items-center text-xs bg-green-100 text-green-700 px-2 py-0.5 rounded">
                          <CheckCircle className="w-3 h-3 mr-1" />
                          Active
                        </span>
                      )}
                    </div>

                    <div className="text-xs text-gray-600 space-y-1">
                      <div className="flex items-center space-x-2">
                        <Clock className="w-3 h-3" />
                        <span>{formatDate(model.timestamp)}</span>
                      </div>
                      
                      <div className="grid grid-cols-2 gap-2 mt-2">
                        <div>
                          <span className="text-gray-500">Test Acc:</span>{' '}
                          <span className="font-semibold">
                            {(model.metrics.test_accuracy * 100).toFixed(2)}%
                          </span>
                        </div>
                        <div>
                          <span className="text-gray-500">Train Acc:</span>{' '}
                          <span className="font-semibold">
                            {(model.metrics.train_accuracy * 100).toFixed(2)}%
                          </span>
                        </div>
                        <div>
                          <span className="text-gray-500">F1:</span>{' '}
                          <span className="font-semibold">
                            {(model.metrics.test_f1_macro * 100).toFixed(2)}%
                          </span>
                        </div>
                        <div>
                          <span className="text-gray-500">Type:</span>{' '}
                          <span className="font-semibold text-xs">
                            {model.model_type}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Trend indicator */}
                  {trend && (
                    <div className={`flex items-center space-x-1 ${trend.color} ml-2`}>
                      <TrendIcon className="w-4 h-4" />
                      <span className="text-xs font-semibold">{trend.text}</span>
                    </div>
                  )}
                </div>

                {/* Expanded details */}
                {selectedModel === model.model_id && (
                  <div className="mt-3 pt-3 border-t border-gray-200 text-xs space-y-1">
                    <div className="font-semibold text-gray-700 mb-2">Full Metrics:</div>
                    <div className="grid grid-cols-2 gap-1 text-gray-600">
                      <div>Precision: {(model.metrics.test_f1_macro * 100).toFixed(2)}%</div>
                      <div>Recall: {(model.metrics.test_f1_macro * 100).toFixed(2)}%</div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* Footer */}
      <div className="text-xs text-gray-500 pt-3 border-t">
        <p>
          💡 Click on a version to see detailed metrics
        </p>
      </div>
    </div>
  );
};

export default ModelVersionTimeline;