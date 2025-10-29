// Feature ranges based on Iris dataset statistics
export const FEATURE_RANGES = {
    'sepal length (cm)': { min: 4.0, max: 8.0, typical: [5.0, 7.0] },
    'sepal width (cm)': { min: 2.0, max: 5.0, typical: [2.5, 3.5] },
    'petal length (cm)': { min: 1.0, max: 7.0, typical: [1.0, 6.0] },
    'petal width (cm)': { min: 0.1, max: 3.0, typical: [0.1, 2.5] },
  };
  
  export const validateFeature = (name, value) => {
    const range = FEATURE_RANGES[name];
    
    if (!range) {
      return { valid: false, message: 'Unknown feature' };
    }
    
    const numValue = parseFloat(value);
    
    if (isNaN(numValue)) {
      return { valid: false, message: 'Must be a number' };
    }
    
    if (numValue < range.min) {
      return { valid: false, message: `Too low (min: ${range.min})` };
    }
    
    if (numValue > range.max) {
      return { valid: false, message: `Too high (max: ${range.max})` };
    }
    
    // Check if within typical range for warning
    const [typicalMin, typicalMax] = range.typical;
    const isTypical = numValue >= typicalMin && numValue <= typicalMax;
    
    return {
      valid: true,
      isTypical,
      message: isTypical ? 'Valid' : 'Unusual value',
    };
  };
  
  export const validateAllFeatures = (features) => {
    const results = {};
    let allValid = true;
    
    Object.entries(features).forEach(([name, value]) => {
      results[name] = validateFeature(name, value);
      if (!results[name].valid) {
        allValid = false;
      }
    });
    
    return { allValid, results };
  };