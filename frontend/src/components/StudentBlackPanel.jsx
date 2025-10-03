import React from "react";

const StudentBlackPanel = ({ onSwitch }) => {
  return (
    <div className="absolute inset-0 p-8 flex flex-col justify-center">
      <div className="max-w-full">
        <h1 className="text-3xl font-bold mb-4">Student Portal</h1>
        
        <p className="text-gray-300 text-sm mb-6 leading-relaxed">
          Get instant answers to your academic questions through our intelligent AI assistant. 
          Access policies, procedures, and get personalized guidance for your educational journey.
        </p>

        <ul className="space-y-2 mb-6 text-sm">
          <li className="flex items-center text-gray-300">
            <span className="w-1.5 h-1.5 bg-white rounded-full mr-3 flex-shrink-0"></span>
            Interactive AI-powered conversations
          </li>
          <li className="flex items-center text-gray-300">
            <span className="w-1.5 h-1.5 bg-white rounded-full mr-3 flex-shrink-0"></span>
            Academic policy and procedure queries
          </li>
          <li className="flex items-center text-gray-300">
            <span className="w-1.5 h-1.5 bg-white rounded-full mr-3 flex-shrink-0"></span>
            Real-time responses with source citations
          </li>
          <li className="flex items-center text-gray-300">
            <span className="w-1.5 h-1.5 bg-white rounded-full mr-3 flex-shrink-0"></span>
            Multi-department knowledge base
          </li>
          <li className="flex items-center text-gray-300">
            <span className="w-1.5 h-1.5 bg-white rounded-full mr-3 flex-shrink-0"></span>
            Voice input capabilities
          </li>
        </ul>

        <button 
          onClick={onSwitch}
          className="bg-white text-gray-900 px-22 py-2 rounded-lg font-semibold hover:bg-gray-100 transition-colors text-sm cursor-pointer"
        >
          Switch to Admin Portal
        </button>
      </div>
    </div>
  );
};

export default StudentBlackPanel;
