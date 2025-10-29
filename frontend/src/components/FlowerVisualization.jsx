import { useEffect, useState } from 'react';

const FlowerVisualization = ({ features }) => {
  const [animate, setAnimate] = useState(false);

  useEffect(() => {
    setAnimate(true);
    const timer = setTimeout(() => setAnimate(false), 500);
    return () => clearTimeout(timer);
  }, [features]);

  // Extract measurements (default to medium values if empty)
  const sepalLength = parseFloat(features['sepal length (cm)']) || 5.5;
  const sepalWidth = parseFloat(features['sepal width (cm)']) || 3.0;
  const petalLength = parseFloat(features['petal length (cm)']) || 4.0;
  const petalWidth = parseFloat(features['petal width (cm)']) || 1.5;

  // Scale factors for visualization
  const sepalLengthScale = (sepalLength / 8.0) * 80 + 40;
  const sepalWidthScale = (sepalWidth / 5.0) * 60 + 30;
  const petalLengthScale = (petalLength / 7.0) * 70 + 30;
  const petalWidthScale = (petalWidth / 3.0) * 50 + 20;

  return (
    <div className="card bg-gradient-to-br from-green-50 to-blue-50">
      <div className="text-center mb-4">
        <h3 className="text-lg font-bold text-gray-800">Live Preview</h3>
        <p className="text-sm text-gray-600">Flower structure based on your measurements</p>
      </div>

      <svg
        viewBox="0 0 300 300"
        className={`w-full h-auto transition-all duration-500 ${animate ? 'scale-105' : 'scale-100'}`}
      >
        {/* Center of flower */}
        <circle cx="150" cy="150" r="15" fill="#FCD34D" />

        {/* Sepals (outer, green-ish) */}
        {[0, 90, 180, 270].map((angle, i) => {
          const rad = (angle * Math.PI) / 180;
          const x = 150 + Math.cos(rad) * 20;
          const y = 150 + Math.sin(rad) * 20;
          
          return (
            <ellipse
              key={`sepal-${i}`}
              cx={x + Math.cos(rad) * (sepalLengthScale / 2)}
              cy={y + Math.sin(rad) * (sepalLengthScale / 2)}
              rx={sepalWidthScale / 2}
              ry={sepalLengthScale / 2}
              fill="#86EFAC"
              opacity="0.8"
              transform={`rotate(${angle}, ${x + Math.cos(rad) * (sepalLengthScale / 2)}, ${y + Math.sin(rad) * (sepalLengthScale / 2)})`}
              className="transition-all duration-500"
            />
          );
        })}

        {/* Petals (inner, colorful) */}
        {[45, 135, 225, 315].map((angle, i) => {
          const rad = (angle * Math.PI) / 180;
          const x = 150 + Math.cos(rad) * 15;
          const y = 150 + Math.sin(rad) * 15;
          
          return (
            <ellipse
              key={`petal-${i}`}
              cx={x + Math.cos(rad) * (petalLengthScale / 2)}
              cy={y + Math.sin(rad) * (petalLengthScale / 2)}
              rx={petalWidthScale / 2}
              ry={petalLengthScale / 2}
              fill="#F472B6"
              opacity="0.9"
              transform={`rotate(${angle}, ${x + Math.cos(rad) * (petalLengthScale / 2)}, ${y + Math.sin(rad) * (petalLengthScale / 2)})`}
              className="transition-all duration-500"
            />
          );
        })}

        {/* Stem */}
        <line
          x1="150"
          y1="165"
          x2="150"
          y2="280"
          stroke="#22C55E"
          strokeWidth="6"
          strokeLinecap="round"
        />
      </svg>

      {/* Legend */}
      <div className="grid grid-cols-2 gap-3 mt-4 text-xs">
        <div className="flex items-center space-x-2">
          <div className="w-4 h-4 rounded-full bg-green-300"></div>
          <span className="text-gray-700">Sepals: {sepalLength.toFixed(1)} × {sepalWidth.toFixed(1)} cm</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-4 h-4 rounded-full bg-pink-400"></div>
          <span className="text-gray-700">Petals: {petalLength.toFixed(1)} × {petalWidth.toFixed(1)} cm</span>
        </div>
      </div>
    </div>
  );
};

export default FlowerVisualization;