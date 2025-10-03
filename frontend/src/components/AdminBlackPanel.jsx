import React from "react";

const AdminBlackPanel = ({ onSwitch }) => {
  return (
    <div className="absolute inset-0 p-8 flex flex-col justify-center">
      <h1 className="text-3xl font-bold mb-4">Admin Portal</h1>
      <p className="text-gray-300 text-sm mb-6 leading-relaxed">
        Manage and upload institutional documents securely. Enhance the knowledge
        base and manage system administration with ease.
      </p>
      <ul className="space-y-2 mb-6 text-sm">
        <li>• Secure PDF document upload</li>
        <li>• Drag & drop file interface</li>
        <li>• Automatic text processing & chunking</li>
        <li>• Real-time upload progress</li>
        <li>• Knowledge base management</li>
        <li>• Document validation & error handling</li>
      </ul>
      <button
        onClick={onSwitch}
        className="bg-white text-gray-900 px-6 py-2 rounded-lg font-semibold hover:bg-gray-100 transition-colors text-sm cursor-pointer"
      >
        Switch to Student Portal
      </button>
    </div>
  );
};

export default AdminBlackPanel;
