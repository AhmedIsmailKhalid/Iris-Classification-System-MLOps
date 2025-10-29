import { useState, useEffect } from 'react';
import Header from './components/Header';
import PredictionForm from './components/PredictionForm';
import FlowerVisualization from './components/FlowerVisualization';
import ResultDisplay from './components/ResultDisplay';
import FeatureContributionChart from './components/FeatureContributionChart';
import PredictionHistory from './components/PredictionHistory';
import ModelPerformanceDashboard from './components/ModelPerformanceDashboard';
import LoadingSpinner from './components/LoadingSpinner';
import ErrorMessage from './components/ErrorMessage';
import { predict } from './services/api';
import { savePrediction } from './utils/storage';

function App() {
  const [features, setFeatures] = useState({
    'sepal length (cm)': '',
    'sepal width (cm)': '',
    'petal length (cm)': '',
    'petal width (cm)': '',
  });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [historyKey, setHistoryKey] = useState(0);

  // Clear error after 5 seconds
  useEffect(() => {
    if (error) {
      const timer = setTimeout(() => setError(null), 5000);
      return () => clearTimeout(timer);
    }
  }, [error]);

  const handlePredict = async (inputFeatures) => {
    try {
      setLoading(true);
      setError(null);
      
      const startTime = performance.now();
      const prediction = await predict(inputFeatures);
      const endTime = performance.now();
      
      // Add latency to result
      const resultWithLatency = {
        ...prediction,
        latency: Math.round(endTime - startTime),
      };
      
      setResult(resultWithLatency);
      setFeatures(inputFeatures);
      
      // Save to history
      savePrediction(inputFeatures, prediction);
      setHistoryKey((prev) => prev + 1); // Trigger history refresh
      
    } catch (err) {
      console.error('Prediction error:', err);
      setError(
        err.response?.data?.detail || 
        err.message || 
        'Failed to get prediction. Please check your connection and try again.'
      );
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  const handleReloadFromHistory = (historyEntry) => {
    console.log('Loading from history:', historyEntry);
    
    // Load features from history
    setFeatures(historyEntry.features);
    
    // Reconstruct the result object from history
    const reconstructedResult = {
      prediction: historyEntry.prediction,
      confidence: historyEntry.confidence,
      probabilities: historyEntry.probabilities,
      feature_contributions: historyEntry.feature_contributions || null,
      model_version: historyEntry.model_version || 'v1.0.0',
      latency: null, // Historical, no latency data
    };
    
    console.log('Reconstructed result:', reconstructedResult);
    
    setResult(reconstructedResult);
    setError(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-pink-50 to-blue-50">
      <Header />
      
      <main className="container mx-auto px-4 py-8">
        {/* Error Message */}
        {error && (
          <div className="mb-6">
            <ErrorMessage message={error} onClose={() => setError(null)} />
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column - Input Form */}
          <div className="lg:col-span-1 space-y-6">
            <PredictionForm 
              onPredict={handlePredict} 
              isLoading={loading}
            />
            
            <FlowerVisualization features={features} />
          </div>

          {/* Middle Column - Results */}
          <div className="lg:col-span-1 space-y-6">
            {loading ? (
              <div className="card">
                <LoadingSpinner message="Analyzing flower..." />
              </div>
            ) : result ? (
              <>
                <ResultDisplay result={result} />
                <FeatureContributionChart 
                  contributions={result.feature_contributions} 
                />
              </>
            ) : (
              <div className="card bg-gradient-to-br from-purple-50 to-pink-50 text-center py-12">
                <div className="text-6xl mb-4">🌸</div>
                <h3 className="text-xl font-bold text-gray-800 mb-2">
                  Ready to Classify
                </h3>
                <p className="text-gray-600">
                  Enter flower measurements and click "Classify Flower" to get started
                </p>
              </div>
            )}
          </div>

          {/* Right Column - History & Model Info */}
          <div className="lg:col-span-1 space-y-6">
            <ModelPerformanceDashboard />
            <PredictionHistory 
              key={historyKey}
              onReload={handleReloadFromHistory}
            />
          </div>
        </div>

        {/* Footer */}
        <footer className="mt-12 text-center text-gray-600 text-sm pb-8">
          <div className="border-t border-gray-200 pt-6">
            <p className="mb-2">
              Built with <span className="text-red-500">♥</span> using React, FastAPI, Docker, and GitHub Actions
            </p>
            <div className="flex items-center justify-center space-x-4 text-xs">
              <span>Tech Stack: React + Vite</span>
              <span>•</span>
              <span>FastAPI + scikit-learn</span>
              <span>•</span>
              <span>SHAP Explanations</span>
            </div>
            <p className="mt-4 text-xs text-gray-500">
              © 2025 Ahmed Ismail Khalid • MIT License
            </p>
          </div>
        </footer>
      </main>
    </div>
  );
}

export default App;