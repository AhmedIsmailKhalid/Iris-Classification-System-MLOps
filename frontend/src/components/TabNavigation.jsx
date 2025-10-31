const TabNavigation = ({ activeTab, setActiveTab }) => {
    const tabs = [
      { id: 'prediction', label: '🎯 Prediction', icon: '🌸' },
      { id: 'mlops', label: '⚙️ MLOps Dashboard', icon: '📊' },
    ];
  
    return (
      <div className="bg-white border-b border-gray-200 sticky top-0 z-10 shadow-sm">
        <div className="max-w-7xl mx-auto px-4">
          <div className="flex space-x-1">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-6 py-4 font-medium transition-all relative ${
                  activeTab === tab.id
                    ? 'text-purple-600 border-b-2 border-purple-600'
                    : 'text-gray-600 hover:text-gray-800 hover:bg-gray-50'
                }`}
              >
                <span className="flex items-center space-x-2">
                  <span>{tab.icon}</span>
                  <span>{tab.label}</span>
                </span>
                {activeTab === tab.id && (
                  <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-gradient-to-r from-purple-600 to-pink-600"></div>
                )}
              </button>
            ))}
          </div>
        </div>
      </div>
    );
  };
  
  export default TabNavigation;