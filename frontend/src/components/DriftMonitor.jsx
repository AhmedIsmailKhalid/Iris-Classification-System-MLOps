import { useState, useEffect } from 'react';
import { Activity, AlertTriangle, CheckCircle, RefreshCw, Zap } from 'lucide-react';
import { getDataStats, generateData, checkDrift } from '../services/api';
import LoadingSpinner from './LoadingSpinner';

const DriftMonitor = () => {
  const [stats, setStats] = useState(null);
  const [driftResult, setDriftResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [checking, setChecking] = useState(false);
  const [generating, setGenerating] = useState(false);

  useEffect(() => {
    loadStats();
  }, []);

  const loadStats = async () => {
    try {
      const data = await getDataStats();
      setStats(data);
    } catch (error) {
      console.error('Failed to load stats:', error);
    }
  };

  const handleSimulateDrift = async () => {
    try {
      setGenerating(true);
      
      // Generate drifted data
      await generateData({
        n_samples: 50,
        data_type: 'drifted',
        drift_type: 'shift',
        drift_magnitude: 2.5,
      });

      // Refresh stats
      await loadStats();
      
      // Auto-check drift
      await handleCheckDrift();
      
    } catch (error) {
      console.error('Failed to simulate drift:', error);
    } finally {
      setGenerating(false);
    }
  };

  const handleCheckDrift = async () => {
    try {
      setChecking(true);
      const result = await checkDrift();
      setDriftResult(result);
      await loadStats();
    } catch (error) {
      console.error('Failed to check drift:', error);
    } finally {
      setChecking(false);
    }
  };

  const getDriftStatusColor = () => {
    if (!driftResult) return 'bg-gray-100 text-gray-700';
    if (driftResult.drift_detected) {
      if (driftResult.drift_severity >= 0.5) return 'bg-red-100 text-red-700';
      return 'bg-yellow-100 text-yellow-700';
    }
    return 'bg-green-100 text-green-700';
  };

  const getDriftStatusIcon = () => {
    if (!driftResult) return <Activity className="w-5 h-5" />;
    if (driftResult.drift_detected) return <AlertTriangle className="w-5 h-5" />;
    return <CheckCircle className="w-5 h-5" />;
  };

  if (!stats) {
    return (
      <div className="card">
        <LoadingSpinner message="Loading drift monitor..." />
      </div>
    );
  }

  return (
    <div className="card space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Activity className="w-6 h-6 text-purple-600" />
          <h3 className="text-lg font-bold text-gray-800">Drift Monitor</h3>
        </div>
        <button
          onClick={loadStats}
          className="p-2 text-gray-600 hover:text-purple-600 transition-colors"
          title="Refresh"
        >
          <RefreshCw className="w-5 h-5" />
        </button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 gap-3">
        <div className="bg-blue-50 rounded-lg p-3 text-center">
          <div className="text-2xl font-bold text-blue-700">
            {stats.new_data_samples}
          </div>
          <div className="text-xs text-gray-600 mt-1">New Samples</div>
        </div>
        
        <div className="bg-purple-50 rounded-lg p-3 text-center">
          <div className="text-2xl font-bold text-purple-700">
            {stats.total_predictions}
          </div>
          <div className="text-xs text-gray-600 mt-1">Total Predictions</div>
        </div>
      </div>

      {/* Drift Status */}
      {driftResult && (
        <div className={`rounded-lg p-4 ${getDriftStatusColor()} animate-fade-in`}>
          <div className="flex items-start space-x-3">
            {getDriftStatusIcon()}
            <div className="flex-1">
              <div className="font-semibold mb-1">
                {driftResult.drift_detected ? 'Drift Detected!' : 'No Drift Detected'}
              </div>
              
              {driftResult.drift_detected && (
                <div className="text-sm space-y-1">
                  <div>
                    Severity: {(driftResult.drift_severity * 100).toFixed(0)}%
                  </div>
                  <div>
                    Affected: {driftResult.drifted_features.join(', ')}
                  </div>
                  <div className="font-semibold mt-2">
                    Recommendation: {driftResult.recommendation.replace('_', ' ')}
                  </div>
                </div>
              )}
              
              {!driftResult.drift_detected && (
                <div className="text-sm">
                  Data distribution is stable
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Action Buttons */}
      <div className="space-y-2">
        <button
          onClick={handleSimulateDrift}
          disabled={generating}
          className="w-full btn-primary flex items-center justify-center space-x-2"
        >
          <Zap className="w-5 h-5" />
          <span>{generating ? 'Generating...' : '🌊 Simulate Data Drift'}</span>
        </button>

        <button
          onClick={handleCheckDrift}
          disabled={checking || stats.new_data_samples < 30}
          className="w-full btn-secondary flex items-center justify-center space-x-2"
        >
          <Activity className="w-5 h-5" />
          <span>
            {checking ? 'Checking...' : 'Check for Drift'}
          </span>
        </button>
        
        {stats.new_data_samples < 30 && (
          <p className="text-xs text-gray-500 text-center">
            Need at least 30 samples to check drift
          </p>
        )}
      </div>

      {/* Info */}
      <div className="text-xs text-gray-500 pt-3 border-t">
        <p className="mb-1">
          💡 <strong>Simulate Drift:</strong> Generates synthetic data with distribution shift
        </p>
        <p>
          📊 <strong>Check Drift:</strong> Runs statistical tests (KS test, PSI) to detect changes
        </p>
      </div>
    </div>
  );
};

export default DriftMonitor;