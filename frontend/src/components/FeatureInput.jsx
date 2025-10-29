import { AlertCircle, CheckCircle2, AlertTriangle } from 'lucide-react';
import { validateFeature } from '../utils/validation';

const FeatureInput = ({ name, value, onChange, onBlur }) => {
  const validation = validateFeature(name, value);
  
  const getInputClasses = () => {
    if (!value) return 'input-field';
    
    if (!validation.valid) {
      return 'input-field border-red-500 focus:ring-red-500';
    }
    
    if (!validation.isTypical) {
      return 'input-field border-yellow-500 focus:ring-yellow-500';
    }
    
    return 'input-field border-green-500 focus:ring-green-500';
  };
  
  const getIcon = () => {
    if (!value) return null;
    
    if (!validation.valid) {
      return <AlertCircle className="w-5 h-5 text-red-500" />;
    }
    
    if (!validation.isTypical) {
      return <AlertTriangle className="w-5 h-5 text-yellow-500" />;
    }
    
    return <CheckCircle2 className="w-5 h-5 text-green-500" />;
  };

  return (
    <div className="space-y-2">
      <label className="label">
        {name.charAt(0).toUpperCase() + name.slice(1)}
      </label>
      
      <div className="relative">
        <input
          type="number"
          step="0.1"
          value={value}
          onChange={(e) => onChange(name, e.target.value)}
          onBlur={onBlur}
          className={getInputClasses()}
          placeholder="Enter value"
        />
        
        <div className="absolute right-3 top-1/2 -translate-y-1/2">
          {getIcon()}
        </div>
      </div>
      
      {value && (
        <p className={`text-xs ${
          !validation.valid ? 'text-red-600' : 
          !validation.isTypical ? 'text-yellow-600' : 
          'text-green-600'
        }`}>
          {validation.message}
        </p>
      )}
    </div>
  );
};

export default FeatureInput;