import { useState, useEffect } from 'react';
import { Activity, AlertTriangle, CheckCircle, RefreshCw, Zap, Rocket, Clock, ExternalLink } from 'lucide-react';
import { getDataStats, generateData, checkDrift, triggerRetraining, getWorkflowStatus } from '../services/api';
import LoadingSpinner from './LoadingSpinner';

const DriftMonitor = () => {
  const [stats, setStats] = useState(null);
  const [driftResult, setDriftResult] = useState(null);
  const [workflowStatus, setWorkflowStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [checking, setChecking] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [retraining, setRetraining] = useState(false);

  useEffect(() => {
    loadStats();
    loadWorkflowStatus();
    
    // Poll workflow status every 30 seconds
    const interval = setInterval(loadWorkflowStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  const loadStats = async () => {
    try {
      const data = await getDataStats();
      setStats(data);
    } catch (error) {
      console.error('Failed to load stats:', error);
    }
  };

  const loadWorkflowStatus = async () => {
    try {
      const data = await getWorkflowStatus();
      setWorkflowStatus(data);
    } catch (error) {
      console.error('Failed to load workflow status:', error);
    }
  };

  const handleSimulateDrift = async () => {
    try {
      setGenerating(true);
      
      await generateData({
        n_samples: 50,
        data_type: 'drifted',
        drift_type: 'shift',
        drift_magnitude: 2.5,
      });

      await loadStats();
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

  const handleTriggerRetraining = async () => {
    try {
      setRetraining(true);
      const result = await triggerRetraining(false);
      
      if (result.success) {
        alert(`✅ ${result.message}\n\nEstimated time: ${result.estimated_time || '3-5 minutes'}\n\nCheck GitHub Actions for live progress!`);
        
        // Start polling for updates
        const pollInterval = setInterval(async () => {
          await loadWorkflowStatus();
          await loadStats();
        }, 10000); // Every 10 seconds
        
        // Stop polling after 5 minutes
        setTimeout(() => clearInterval(pollInterval), 300000);
      } else {
        alert(`❌ ${result.message}\n\n${result.error || ''}`);
      }
      
    } catch (error) {
      console.error('Failed to trigger retraining:', error);
      alert('❌ Failed to trigger retraining. Check console for details.');
    } finally {
      setRetraining(false);
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

  const getWorkflowStatusColor = (status, conclusion) => {
    if (status === 'in_progress' || status === 'queued') return 'text-blue-600';
    if (status === 'completed' && conclusion === 'success') return 'text-green-600';
    if (status === 'completed' && conclusion === 'failure') return 'text-red-600';
    return 'text-gray-600';
  };

  const getWorkflowStatusText = (status, conclusion) => {
    if (status === 'in_progress') return '🔄 Running...';
    if (status === 'queued') return '⏳ Queued';
    if (status === 'completed' && conclusion === 'success') return '✅ Success';
    if (status === 'completed' && conclusion === 'failure') return '❌ Failed';
    return status;
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
          <h3 className="text-lg font-bold text-gray-800">MLOps Pipeline</h3>
        </div>
        <button
          onClick={() => {
            loadStats();
            loadWorkflowStatus();
          }}
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
          <div className="text-xs text-gray-600 mt-1">Predictions</div>
        </div>
      </div>

      {/* Latest Workflow Status */}
      {workflowStatus?.success && workflowStatus.runs?.length > 0 && (
        <div className="bg-gray-50 rounded-lg p-3">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-semibold text-gray-700">Latest Workflow</span>
            <a 
              href={workflowStatus.runs[0].html_url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs text-blue-600 hover:text-blue-700 flex items-center gap-1"
            >
              View <ExternalLink className="w-3 h-3" />
            </a>
          </div>
          <div className="flex items-center justify-between">
            <span className={`text-sm font-medium ${getWorkflowStatusColor(workflowStatus.runs[0].status, workflowStatus.runs[0].conclusion)}`}>
              {getWorkflowStatusText(workflowStatus.runs[0].status, workflowStatus.runs[0].conclusion)}
            </span>
            <span className="text-xs text-gray-500">
              Run #{workflowStatus.runs[0].run_number}
            </span>
          </div>
        </div>
      )}

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
                  <div className="text-xs">
                    Affected: {driftResult.drifted_features.join(', ')}
                  </div>
                  <div className="font-semibold mt-2 text-xs">
                    📋 {driftResult.recommendation.replace('_', ' ').toUpperCase()}
                  </div>
                </div>
              )}
              
              {!driftResult.drift_detected && (
                <div className="text-sm">
                  Data distribution is stable ✨
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
          <span>{generating ? 'Generating...' : ' Simulate Data Drift'}</span>
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

        <button
          onClick={handleTriggerRetraining}
          disabled={retraining || stats.new_data_samples < 30 || !driftResult?.drift_detected}
          className="w-full bg-gradient-to-r from-green-600 to-emerald-600 text-white py-2 px-4 rounded-lg font-medium hover:from-green-700 hover:to-emerald-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center justify-center space-x-2"
        >
          <Rocket className="w-5 h-5" />
          <span>
            {retraining ? 'Triggering...' : ' Trigger Retraining'}
          </span>
        </button>
        
        {stats.new_data_samples < 30 && (
          <p className="text-xs text-gray-500 text-center">
            Need at least 30 samples
          </p>
        )}

        {stats.new_data_samples >= 30 && !driftResult?.drift_detected && (
          <p className="text-xs text-gray-500 text-center">
            Run drift check first
          </p>
        )}
      </div>

      {/* Info */}
      <div className="text-xs text-gray-500 pt-3 border-t space-y-1">
        <p>
          <strong>Simulate Drift:</strong> Generates data with distribution shift
        </p>
        <p>
          <strong>Check Drift:</strong> Runs KS test + PSI analysis
        </p>
        <p>
          <strong>Trigger Retraining:</strong> Starts GitHub Actions workflow
        </p>
      </div>
    </div>
  );
};

export default DriftMonitor;