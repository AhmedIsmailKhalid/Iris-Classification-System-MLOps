import { useState } from 'react';
import { Send, RotateCcw } from 'lucide-react';
import FeatureInput from './FeatureInput';
import QuickFillDropdown from './QuickFillDropdown';
import { validateAllFeatures } from '../utils/validation';

const PredictionForm = ({ onPredict, isLoading }) => {
  const [features, setFeatures] = useState({
    'sepal length (cm)': '',
    'sepal width (cm)': '',
    'petal length (cm)': '',
    'petal width (cm)': '',
  });

  const handleFeatureChange = (name, value) => {
    setFeatures((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleQuickFill = (exampleFeatures) => {
    setFeatures(exampleFeatures);
  };

  const handleReset = () => {
    setFeatures({
      'sepal length (cm)': '',
      'sepal width (cm)': '',
      'petal length (cm)': '',
      'petal width (cm)': '',
    });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    
    // Validate all features
    const { allValid } = validateAllFeatures(features);
    
    if (!allValid) {
      return;
    }

    // Convert to numbers
    const numericFeatures = Object.entries(features).reduce((acc, [key, value]) => {
      acc[key] = parseFloat(value);
      return acc;
    }, {});

    onPredict(numericFeatures);
  };

  const { allValid } = validateAllFeatures(features);
  const hasAllValues = Object.values(features).every((v) => v !== '');

  return (
    <div className="card space-y-6">
      <div>
        <h2 className="text-xl font-bold text-gray-800 mb-2">
          Enter Measurements
        </h2>
        <p className="text-sm text-gray-600">
          Provide the flower's measurements to get a prediction
        </p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Quick Fill Dropdown */}
        <QuickFillDropdown onFill={handleQuickFill} />

        <div className="border-t pt-4"></div>

        {/* Feature Inputs */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {Object.keys(features).map((featureName) => (
            <FeatureInput
              key={featureName}
              name={featureName}
              value={features[featureName]}
              onChange={handleFeatureChange}
            />
          ))}
        </div>

        {/* Action Buttons */}
        <div className="flex flex-col sm:flex-row gap-3 pt-4">
          <button
            type="submit"
            disabled={!hasAllValues || !allValid || isLoading}
            className="btn-primary flex-1 flex items-center justify-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Send className="w-5 h-5" />
            <span>{isLoading ? 'Classifying...' : 'Classify Flower'}</span>
          </button>

          <button
            type="button"
            onClick={handleReset}
            disabled={isLoading}
            className="btn-secondary flex items-center justify-center space-x-2"
          >
            <RotateCcw className="w-5 h-5" />
            <span>Reset</span>
          </button>
        </div>
      </form>
    </div>
  );
};

export default PredictionForm;