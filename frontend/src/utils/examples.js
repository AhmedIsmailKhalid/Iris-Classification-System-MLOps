// Predefined example flower measurements
export const EXAMPLE_FLOWERS = {
    setosa: {
      name: 'Typical Setosa',
      features: {
        'sepal length (cm)': 5.1,
        'sepal width (cm)': 3.5,
        'petal length (cm)': 1.4,
        'petal width (cm)': 0.2,
      },
      description: 'Small petals, distinctive setosa characteristics',
    },
    versicolor: {
      name: 'Typical Versicolor',
      features: {
        'sepal length (cm)': 6.4,
        'sepal width (cm)': 3.2,
        'petal length (cm)': 4.5,
        'petal width (cm)': 1.5,
      },
      description: 'Medium-sized petals, intermediate features',
    },
    virginica: {
      name: 'Typical Virginica',
      features: {
        'sepal length (cm)': 6.3,
        'sepal width (cm)': 3.3,
        'petal length (cm)': 6.0,
        'petal width (cm)': 2.5,
      },
      description: 'Large petals, robust virginica traits',
    },
    random: {
      name: 'Random Sample',
      features: null, // Will be generated
      description: 'Randomly generated valid measurements',
    },
  };
  
  export const generateRandomFeatures = () => {
    return {
      'sepal length (cm)': (Math.random() * 3.5 + 4.5).toFixed(1),
      'sepal width (cm)': (Math.random() * 2.0 + 2.5).toFixed(1),
      'petal length (cm)': (Math.random() * 5.0 + 1.5).toFixed(1),
      'petal width (cm)': (Math.random() * 2.0 + 0.5).toFixed(1),
    };
  };
  
  export const getExampleFeatures = (exampleType) => {
    if (exampleType === 'random') {
      return generateRandomFeatures();
    }
    return EXAMPLE_FLOWERS[exampleType]?.features || null;
  };