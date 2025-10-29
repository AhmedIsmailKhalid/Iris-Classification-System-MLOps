import { CheckCircle, TrendingUp, Percent } from 'lucide-react';

const ResultDisplay = ({ result }) => {
  if (!result) return null;

  const { prediction, confidence, probabilities } = result;

  const getSpeciesEmoji = (species) => {
    const emojis = {
      setosa: '🌸',
      versicolor: '🌺',
      virginica: '🌷',
    };
    return emojis[species.toLowerCase()] || '🌼';
  };

  const getConfidenceColor = (conf) => {
    if (conf >= 0.9) return 'text-green-600 bg-green-50';
    if (conf >= 0.7) return 'text-yellow-600 bg-yellow-50';
    return 'text-orange-600 bg-orange-50';
  };

  return (
    <div className="card space-y-6 animate-slide-up">
      {/* Header */}
      <div className="flex items-center space-x-3">
        <CheckCircle className="w-8 h-8 text-green-500" />
        <div>
          <h2 className="text-xl font-bold text-gray-800">Prediction Result</h2>
          <p className="text-sm text-gray-600">Classification complete</p>
        </div>
      </div>

      {/* Main Prediction */}
      <div className="bg-gradient-to-br from-purple-50 to-pink-50 rounded-xl p-6 border border-purple-200">
        <div className="text-center">
          <div className="text-6xl mb-3">{getSpeciesEmoji(prediction)}</div>
          <h3 className="text-3xl font-bold text-gray-800 mb-2">
            {prediction.charAt(0).toUpperCase() + prediction.slice(1)}
          </h3>
          <p className="text-gray-600">Iris Species</p>
        </div>
      </div>

      {/* Confidence Score */}
      <div className={`rounded-lg p-4 ${getConfidenceColor(confidence)}`}>
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center space-x-2">
            <TrendingUp className="w-5 h-5" />
            <span className="font-semibold">Confidence Score</span>
          </div>
          <span className="text-2xl font-bold">{(confidence * 100).toFixed(1)}%</span>
        </div>
        
        <div className="w-full bg-white/50 rounded-full h-3 overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-purple-500 to-pink-500 rounded-full transition-all duration-1000"
            style={{ width: `${confidence * 100}%` }}
          />
        </div>
      </div>

      {/* Probability Distribution */}
      <div className="space-y-3">
        <div className="flex items-center space-x-2 text-gray-700">
          <Percent className="w-5 h-5" />
          <h4 className="font-semibold">Class Probabilities</h4>
        </div>

        <div className="space-y-2">
          {Object.entries(probabilities)
            .sort(([, a], [, b]) => b - a)
            .map(([species, prob]) => (
              <div key={species} className="space-y-1">
                <div className="flex justify-between text-sm">
                  <span className="font-medium text-gray-700 capitalize">
                    {species}
                  </span>
                  <span className="text-gray-600">
                    {(prob * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2 overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-1000 ${
                      species === prediction.toLowerCase()
                        ? 'bg-gradient-to-r from-purple-500 to-pink-500'
                        : 'bg-gray-400'
                    }`}
                    style={{ width: `${prob * 100}%` }}
                  />
                </div>
              </div>
            ))}
        </div>
      </div>

      {/* Model Version */}
      <div className="text-xs text-gray-500 text-center pt-4 border-t">
        Model Version: {result.model_version || 'v1.0.0'} • 
        Latency: {result.latency ? `${result.latency}ms` : 'N/A'}
      </div>
    </div>
  );
};

export default ResultDisplay;