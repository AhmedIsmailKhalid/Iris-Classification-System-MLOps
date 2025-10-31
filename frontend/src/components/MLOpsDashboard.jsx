import { useState, useEffect } from 'react';
import WorkflowStatusViewer from './WorkflowStatusViewer';
import DriftMonitor from './DriftMonitor';
import ModelVersionTimeline from './ModelVersionTimeline';
import DriftVisualization from './DriftVisualization';
import ArchitectureDiagram from './ArchitectureDiagram';

const MLOpsDashboard = () => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-pink-50 to-blue-50 py-8">
      <div className="max-w-7xl mx-auto px-4">
        {/* Dashboard Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-2">
            MLOps Dashboard
          </h1>
          <p className="text-gray-600">
            Monitor drift detection, automated retraining, and GitHub Actions workflows
          </p>
        </div>

        <ArchitectureDiagram />

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left Column - Drift & Data + Model Version */}
          <div className="space-y-6">
            {/* <DriftMonitor /> */}

            {/* <div className="card">
              <h3 className="text-lg font-bold text-gray-800 mb-4">
                📈 Model Version Timeline
              </h3>
              <p className="text-gray-500 text-sm">Coming soon...</p>
            </div> */}
            <ModelVersionTimeline />

            {/* <div className="card">
              <h3 className="text-lg font-bold text-gray-800 mb-4">
                📊 Drift Visualization
              </h3>
              <p className="text-gray-500 text-sm">Coming soon...</p>
            </div> */}
            <DriftVisualization />

          </div>

          {/* Right Column - Workflows */}
          <div className="space-y-6">
            <WorkflowStatusViewer />

            {/* Placeholder for Drift Visualization */}
            {/* <div className="card">
              <h3 className="text-lg font-bold text-gray-800 mb-4">
                📊 Drift Visualization
              </h3>
              <p className="text-gray-500 text-sm">Coming soon...</p>
            </div> */}
          </div>
        </div>
      </div>
    </div>
  );
};

export default MLOpsDashboard;