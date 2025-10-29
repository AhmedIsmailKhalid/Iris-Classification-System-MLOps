import { BarChart3, Info } from 'lucide-react';
import { useState } from 'react';

const FeatureContributionChart = ({ contributions }) => {
  const [showExplanation, setShowExplanation] = useState(false);

  if (!contributions) {
    return (
      <div className="card bg-gray-50">
        <p className="text-gray-500 text-center">
          Feature contributions will appear after prediction
        </p>
      </div>
    );
  }

  // Get sorted contributions
  const sortedContributions = Object.entries(contributions).sort(
    ([, a], [, b]) => Math.abs(b) - Math.abs(a)
  );

  const maxAbsValue = Math.max(...sortedContributions.map(([, val]) => Math.abs(val)));

  return (
    <div className="card space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <BarChart3 className="w-6 h-6 text-purple-600" />
          <h3 className="text-lg font-bold text-gray-800">Feature Importance</h3>
        </div>
        <button
          onClick={() => setShowExplanation(!showExplanation)}
          className="text-purple-600 hover:text-purple-700 transition-colors"
        >
          <Info className="w-5 h-5" />
        </button>
      </div>

      {/* Explanation */}
      {showExplanation && (
        <div className="bg-purple-50 border border-purple-200 rounded-lg p-4 text-sm text-gray-700 animate-fade-in">
          <p>
            <strong>SHAP values</strong> show how much each feature contributed to this specific prediction.
            Positive values (red) push toward the predicted class, while negative values (blue) push away.
            Longer bars = stronger influence.
          </p>
        </div>
      )}

      {/* Contributions Chart */}
      <div className="space-y-3">
        {sortedContributions.map(([feature, value]) => {
          const isPositive = value >= 0;
          const percentage = (Math.abs(value) / maxAbsValue) * 100;
          
          return (
            <div key={feature} className="space-y-1">
              <div className="flex justify-between items-center text-sm">
                <span className="font-medium text-gray-700 capitalize">
                  {feature.replace(' (cm)', '')}
                </span>
                <span className={`font-semibold ${isPositive ? 'text-red-600' : 'text-blue-600'}`}>
                  {isPositive ? '+' : ''}{value.toFixed(3)}
                </span>
              </div>
              
              <div className="relative w-full h-8 bg-gray-100 rounded-lg overflow-hidden">
                {/* Center line */}
                <div className="absolute left-1/2 top-0 bottom-0 w-0.5 bg-gray-300 z-10"></div>
                
                {/* Bar */}
                <div
                  className={`absolute top-0 bottom-0 transition-all duration-700 ${
                    isPositive
                      ? 'bg-gradient-to-r from-red-400 to-red-600 left-1/2'
                      : 'bg-gradient-to-l from-blue-400 to-blue-600 right-1/2'
                  }`}
                  style={{
                    width: `${percentage / 2}%`,
                  }}
                >
                  <div className="h-full flex items-center justify-center">
                    <span className="text-xs font-semibold text-white px-2">
                      {Math.abs(value) > 0.01 ? Math.abs(value).toFixed(2) : ''}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Legend */}
      <div className="flex items-center justify-center space-x-6 pt-2 text-xs text-gray-600 border-t">
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 bg-gradient-to-r from-red-400 to-red-600 rounded"></div>
          <span>Increases probability</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 bg-gradient-to-l from-blue-400 to-blue-600 rounded"></div>
          <span>Decreases probability</span>
        </div>
      </div>
    </div>
  );
};

export default FeatureContributionChart;