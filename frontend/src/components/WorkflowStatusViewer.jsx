import { useState, useEffect } from 'react';
import { Activity, CheckCircle, XCircle, Clock, ExternalLink, RefreshCw, GitBranch } from 'lucide-react';
import { getWorkflowStatus } from '../services/api';

const WorkflowStatusViewer = () => {
  const [workflows, setWorkflows] = useState([]);
  const [loading, setLoading] = useState(true);
  const [autoRefresh, setAutoRefresh] = useState(true);

  useEffect(() => {
    loadWorkflows();

    // Auto-refresh every 10 seconds if enabled
    let interval;
    if (autoRefresh) {
      interval = setInterval(loadWorkflows, 10000);
    }

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [autoRefresh]);

  const loadWorkflows = async () => {
    try {
      const data = await getWorkflowStatus();
      if (data.success && data.runs) {
        setWorkflows(data.runs);
      }
    } catch (error) {
      console.error('Failed to load workflows:', error);
    } finally {
      setLoading(false);
    }
  };

  const getStatusIcon = (status, conclusion) => {
    if (status === 'in_progress' || status === 'queued') {
      return <Activity className="w-5 h-5 text-blue-600 animate-spin" />;
    }
    if (status === 'completed') {
      if (conclusion === 'success') {
        return <CheckCircle className="w-5 h-5 text-green-600" />;
      }
      if (conclusion === 'failure') {
        return <XCircle className="w-5 h-5 text-red-600" />;
      }
    }
    return <Clock className="w-5 h-5 text-gray-400" />;
  };

  const getStatusColor = (status, conclusion) => {
    if (status === 'in_progress') return 'bg-blue-50 border-blue-200';
    if (status === 'queued') return 'bg-yellow-50 border-yellow-200';
    if (status === 'completed' && conclusion === 'success') return 'bg-green-50 border-green-200';
    if (status === 'completed' && conclusion === 'failure') return 'bg-red-50 border-red-200';
    return 'bg-gray-50 border-gray-200';
  };

  const getStatusText = (status, conclusion) => {
    if (status === 'in_progress') return 'Running';
    if (status === 'queued') return 'Queued';
    if (status === 'completed' && conclusion === 'success') return 'Success';
    if (status === 'completed' && conclusion === 'failure') return 'Failed';
    return status;
  };

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    return `${diffDays}d ago`;
  };

  if (loading) {
    return (
      <div className="card">
        <div className="flex items-center justify-center py-8">
          <Activity className="w-6 h-6 text-purple-600 animate-spin" />
          <span className="ml-2 text-gray-600">Loading workflows...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="card space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <GitBranch className="w-6 h-6 text-purple-600" />
          <h3 className="text-lg font-bold text-gray-800">GitHub Actions</h3>
        </div>
        <div className="flex items-center space-x-2">
          <button
            onClick={() => setAutoRefresh(!autoRefresh)}
            className={`text-xs px-2 py-1 rounded ${
              autoRefresh ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-600'
            }`}
          >
            {autoRefresh ? '🔄 Auto' : '⏸️ Paused'}
          </button>
          <button
            onClick={loadWorkflows}
            className="p-2 text-gray-600 hover:text-purple-600 transition-colors"
            title="Refresh"
          >
            <RefreshCw className="w-5 h-5" />
          </button>
        </div>
      </div>

      {/* Workflows List */}
      {workflows.length === 0 ? (
        <div className="text-center py-8 text-gray-500">
          <GitBranch className="w-12 h-12 mx-auto mb-2 text-gray-300" />
          <p>No workflow runs yet</p>
          <p className="text-xs mt-1">Trigger retraining to see workflows here</p>
        </div>
      ) : (
        <div className="space-y-3">
          {workflows.map((workflow) => (
            <div
              key={workflow.id}
              className={`border rounded-lg p-4 ${getStatusColor(
                workflow.status,
                workflow.conclusion
              )} transition-all`}
            >
              <div className="flex items-start justify-between">
                <div className="flex items-start space-x-3 flex-1">
                  {getStatusIcon(workflow.status, workflow.conclusion)}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center space-x-2 mb-1">
                      <span className="font-semibold text-gray-800">
                        Run #{workflow.run_number}
                      </span>
                      <span
                        className={`text-xs px-2 py-0.5 rounded ${
                          workflow.status === 'in_progress'
                            ? 'bg-blue-100 text-blue-700'
                            : workflow.status === 'completed' && workflow.conclusion === 'success'
                            ? 'bg-green-100 text-green-700'
                            : workflow.status === 'completed' && workflow.conclusion === 'failure'
                            ? 'bg-red-100 text-red-700'
                            : 'bg-gray-100 text-gray-700'
                        }`}
                      >
                        {getStatusText(workflow.status, workflow.conclusion)}
                      </span>
                    </div>
                    <div className="text-sm text-gray-600 space-y-1">
                      <div className="flex items-center space-x-2">
                        <Clock className="w-3 h-3" />
                        <span>{formatDate(workflow.updated_at)}</span>
                      </div>
                      {workflow.status === 'in_progress' && (
                        <div className="flex items-center space-x-2 text-blue-600">
                          <Activity className="w-3 h-3 animate-pulse" />
                          <span className="font-medium">Running...</span>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
                
                  href={workflow.html_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="ml-2 p-2 text-gray-600 hover:text-purple-600 transition-colors"
                  title="View on GitHub"
                <a>
                  <ExternalLink className="w-4 h-4" />
                </a>
              </div>

              {/* Progress bar for running workflows */}
              {workflow.status === 'in_progress' && (
                <div className="mt-3">
                  <div className="h-1 bg-gray-200 rounded-full overflow-hidden">
                    <div className="h-full bg-blue-600 rounded-full animate-pulse w-2/3"></div>
                  </div>
                  <p className="text-xs text-gray-500 mt-1">
                    Estimated time: 3-5 minutes
                  </p>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Footer */}
      <div className="text-xs text-gray-500 pt-3 border-t">
        <p>
          💡 Workflows auto-refresh every 10 seconds when enabled
        </p>
      </div>
    </div>
  );
};

export default WorkflowStatusViewer;