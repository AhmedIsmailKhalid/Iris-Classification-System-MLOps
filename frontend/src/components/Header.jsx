import { Flower2, Github, ExternalLink } from "lucide-react";

const Header = () => {
  return (
    <header className="bg-white shadow-md border-b border-gray-200">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          
          {/* Logo and Title */}
          <div className="flex items-center space-x-3">
            <div className="bg-gradient-to-br from-purple-500 to-pink-500 p-2 rounded-lg">
              <Flower2 className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">
                Iris Flower Classifier
              </h1>
              <p className="text-sm text-gray-600">
                Real-time ML prediction with 90%+ accuracy
              </p>
            </div>
          </div>

          {/* Navigation Links */}
          <div className="flex items-center space-x-4">
            <a
              href="http://localhost:8000/docs"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center space-x-2 text-gray-700 hover:text-purple-600 transition-colors"
            >
              <ExternalLink className="w-4 h-4" />
              <span className="hidden sm:inline">API Docs</span>
            </a>

            <a
              href="https://github.com/AhmedIsmailKhalid/Iris-Classification-System-MLOps"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center space-x-2 text-gray-700 hover:text-purple-600 transition-colors"
            >
              <Github className="w-5 h-5" />
              <span className="hidden sm:inline">GitHub</span>
            </a>
          </div>

        </div>
      </div>
    </header>
  );
};

export default Header;
