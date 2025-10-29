import { Sparkles } from 'lucide-react';
import { EXAMPLE_FLOWERS, getExampleFeatures } from '../utils/examples';

const QuickFillDropdown = ({ onFill }) => {
  const handleSelect = (e) => {
    const exampleType = e.target.value;
    if (!exampleType) return;
    
    const features = getExampleFeatures(exampleType);
    if (features) {
      onFill(features);
    }
    
    // Reset dropdown
    e.target.value = '';
  };

  return (
    <div className="relative">
      <label className="label flex items-center space-x-2">
        <Sparkles className="w-4 h-4 text-purple-600" />
        <span>Quick Fill Examples</span>
      </label>
      
      <select
        onChange={handleSelect}
        className="input-field cursor-pointer"
        defaultValue=""
      >
        <option value="" disabled>
          Choose a preset...
        </option>
        {Object.entries(EXAMPLE_FLOWERS).map(([key, example]) => (
          <option key={key} value={key}>
            {example.name} - {example.description}
          </option>
        ))}
      </select>
    </div>
  );
};

export default QuickFillDropdown;