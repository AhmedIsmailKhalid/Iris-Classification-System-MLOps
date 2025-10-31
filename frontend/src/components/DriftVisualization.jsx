import { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';
import { TrendingUp, AlertTriangle, CheckCircle, RefreshCw, BarChart3 } from 'lucide-react';
import { checkDrift, getDataStats } from '../services/api';

const DriftVisualization = () => {
  const [driftData, setDriftData] = useState(null);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(false);
  const [viewMode, setViewMode] = useState('features'); // 'features', 'metrics', 'radar'

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

  const handleCheckDrift = async () => {
    try {
      setLoading(true);
      const result = await checkDrift();
      
      if (result.drift_detected) {
        setDriftData(result);
      } else {
        setDriftData(null);
        alert('No drift detected. Generate drifted data first.');
      }
      
      await loadStats();
    } catch (error) {
      console.error('Failed to check drift:', error);
      alert('Failed to check drift. Make sure you have enough data (30+ samples).');
    } finally {
      setLoading(false);
    }
  };

  const prepareFeatureComparisonData = () => {
    if (!driftData?.feature_drift_details) return [];

    return driftData.feature_drift_details.map((feature) => ({
      name: feature.feature.replace(' (cm)', ''),
      'Reference Mean': feature.reference_mean.toFixed(2),
      'Current Mean': feature.current_mean.toFixed(2),
      'Difference %': Math.abs(feature.mean_difference_percent).toFixed(1),
    }));
  };

  const prepareDriftMetricsData = () => {
    if (!driftData?.feature_drift_details) return [];

    return driftData.feature_drift_details.map((feature) => ({
      name: feature.feature.replace(' (cm)', ''),
      PSI: feature.psi.toFixed(3),
      'KS Statistic': feature.ks_statistic.toFixed(3),
      'P-Value': feature.ks_pvalue.toFixed(4),
    }));
  };

  const prepareRadarData = () => {
    if (!driftData?.feature_drift_details) return [];

    const features = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'];
    
    return features.map((name, idx) => {
      const detail = driftData.feature_drift_details[idx];
      return {
        feature: name,
        'Drift Score': Math.min(detail.psi * 50, 100), // Scale PSI for visualization
        'Significance': (1 - detail.ks_pvalue) * 100, // Convert p-value to significance %
      };
    });
  };

  const getDriftSeverityColor = (severity) => {
    if (severity >= 0.75) return 'text-red-600';
    if (severity >= 0.5) return 'text-orange-600';
    if (severity >= 0.25) return 'text-yellow-600';
    return 'text-green-600';
  };

  const getDriftSeverityBg = (severity) => {
    if (severity >= 0.75) return 'bg-red-100';
    if (severity >= 0.5) return 'bg-orange-100';
    if (severity >= 0.25) return 'bg-yellow-100';
    return 'bg-green-100';
  };

  return (
    <div className="card space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <BarChart3 className="w-6 h-6 text-purple-600" />
          <h3 className="text-lg font-bold text-gray-800">Drift Visualization</h3>
        </div>
        <button
          onClick={handleCheckDrift}
          disabled={loading || !stats || stats.new_data_samples < 30}
          className="flex items-center space-x-2 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          <span>{loading ? 'Analyzing...' : 'Analyze Drift'}</span>
        </button>
      </div>

      {/* Data requirement notice */}
      {stats && stats.new_data_samples < 30 && (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3 text-sm">
          <div className="flex items-start space-x-2">
            <AlertTriangle className="w-5 h-5 text-yellow-600 flex-shrink-0 mt-0.5" />
            <div>
              <div className="font-semibold text-yellow-800">Insufficient Data</div>
              <div className="text-yellow-700 text-xs mt-1">
                Need at least 30 samples for drift analysis. Current: {stats.new_data_samples}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* No drift data yet */}
      {!driftData && !loading && (
        <div className="text-center py-12 text-gray-500">
          <BarChart3 className="w-16 h-16 mx-auto mb-3 text-gray-300" />
          <p className="font-medium mb-1">No Drift Analysis Yet</p>
          <p className="text-sm">Generate drifted data and click "Analyze Drift" to see visualizations</p>
        </div>
      )}

      {/* Drift detected - show visualizations */}
      {driftData && (
        <>
          {/* Drift Summary */}
          <div className={`rounded-lg p-4 ${getDriftSeverityBg(driftData.drift_severity)}`}>
            <div className="flex items-start space-x-3">
              {driftData.drift_severity >= 0.5 ? (
                <AlertTriangle className="w-6 h-6 text-red-600" />
              ) : (
                <CheckCircle className="w-6 h-6 text-green-600" />
              )}
              <div className="flex-1">
                <div className="font-bold text-gray-800 mb-2">
                  Drift Severity: {(driftData.drift_severity * 100).toFixed(0)}%
                </div>
                <div className="text-sm space-y-1">
                  <div>
                    <span className="font-semibold">Affected Features:</span>{' '}
                    {driftData.drifted_features.length} / {driftData.feature_drift_details.length}
                  </div>
                  <div>
                    <span className="font-semibold">Recommendation:</span>{' '}
                    <span className="uppercase font-bold">
                      {driftData.recommendation.replace('_', ' ')}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* View Mode Selector */}
          <div className="flex space-x-2 border-b">
            <button
              onClick={() => setViewMode('features')}
              className={`px-4 py-2 font-medium transition-colors ${
                viewMode === 'features'
                  ? 'text-purple-600 border-b-2 border-purple-600'
                  : 'text-gray-600 hover:text-gray-800'
              }`}
            >
              📊 Feature Comparison
            </button>
            <button
              onClick={() => setViewMode('metrics')}
              className={`px-4 py-2 font-medium transition-colors ${
                viewMode === 'metrics'
                  ? 'text-purple-600 border-b-2 border-purple-600'
                  : 'text-gray-600 hover:text-gray-800'
              }`}
            >
              📈 Drift Metrics
            </button>
            <button
              onClick={() => setViewMode('radar')}
              className={`px-4 py-2 font-medium transition-colors ${
                viewMode === 'radar'
                  ? 'text-purple-600 border-b-2 border-purple-600'
                  : 'text-gray-600 hover:text-gray-800'
              }`}
            >
              🎯 Radar View
            </button>
          </div>

          {/* Feature Comparison Chart */}
          {viewMode === 'features' && (
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={prepareFeatureComparisonData()}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" angle={-45} textAnchor="end" height={80} />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="Reference Mean" fill="#8b5cf6" />
                  <Bar dataKey="Current Mean" fill="#ec4899" />
                </BarChart>
              </ResponsiveContainer>
              <div className="text-xs text-gray-500 text-center mt-2">
                Comparing reference dataset vs. current data distributions
              </div>
            </div>
          )}

          {/* Drift Metrics Chart */}
          {viewMode === 'metrics' && (
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={prepareDriftMetricsData()}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" angle={-45} textAnchor="end" height={80} />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="PSI" stroke="#8b5cf6" strokeWidth={2} />
                  <Line type="monotone" dataKey="KS Statistic" stroke="#ec4899" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
              <div className="text-xs text-gray-500 text-center mt-2">
                PSI (Population Stability Index) & KS (Kolmogorov-Smirnov) Statistics
              </div>
            </div>
          )}

          {/* Radar Chart */}
          {viewMode === 'radar' && (
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <RadarChart data={prepareRadarData()}>
                  <PolarGrid />
                  <PolarAngleAxis dataKey="feature" />
                  <PolarRadiusAxis angle={90} domain={[0, 100]} />
                  <Radar name="Drift Score" dataKey="Drift Score" stroke="#8b5cf6" fill="#8b5cf6" fillOpacity={0.6} />
                  <Radar name="Significance" dataKey="Significance" stroke="#ec4899" fill="#ec4899" fillOpacity={0.6} />
                  <Legend />
                </RadarChart>
              </ResponsiveContainer>
              <div className="text-xs text-gray-500 text-center mt-2">
                Multi-dimensional drift analysis across all features
              </div>
            </div>
          )}

          {/* Feature Details Table */}
          <div className="mt-4">
            <div className="text-sm font-semibold text-gray-700 mb-2">Detailed Metrics:</div>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead className="bg-gray-50 border-b">
                  <tr>
                    <th className="px-3 py-2 text-left">Feature</th>
                    <th className="px-3 py-2 text-right">PSI</th>
                    <th className="px-3 py-2 text-right">KS Stat</th>
                    <th className="px-3 py-2 text-right">P-Value</th>
                    <th className="px-3 py-2 text-right">Mean Diff %</th>
                    <th className="px-3 py-2 text-center">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y">
                  {driftData.feature_drift_details.map((feature) => (
                    <tr key={feature.feature} className="hover:bg-gray-50">
                      <td className="px-3 py-2 font-medium">{feature.feature.replace(' (cm)', '')}</td>
                      <td className="px-3 py-2 text-right">{feature.psi.toFixed(3)}</td>
                      <td className="px-3 py-2 text-right">{feature.ks_statistic.toFixed(3)}</td>
                      <td className="px-3 py-2 text-right">{feature.ks_pvalue.toFixed(4)}</td>
                      <td className="px-3 py-2 text-right">{feature.mean_difference_percent.toFixed(1)}%</td>
                      <td className="px-3 py-2 text-center">
                        {feature.drift_detected ? (
                          <span className="text-red-600 font-semibold">⚠️ Drift</span>
                        ) : (
                          <span className="text-green-600 font-semibold">✓ OK</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      {/* Info */}
      <div className="text-xs text-gray-500 pt-3 border-t space-y-1">
        <p><strong>PSI:</strong> &lt;0.1 = No change, 0.1-0.2 = Moderate, &gt;0.2 = Significant</p>
        <p><strong>KS Test:</strong> p-value &lt; 0.05 indicates significant distribution difference</p>
      </div>
    </div>
  );
};

export default DriftVisualization;