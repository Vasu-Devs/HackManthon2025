import React from "react";

const AdminWhitePanel = ({
  registrationNumber,
  setRegistrationNumber,
  password,
  setPassword,
  showPassword,
  handleSubmit,
  isLoading = false,
  error = "",
}) => {
  return (
    <div className="absolute inset-0 p-8 flex flex-col justify-center">
      <h2 className="text-3xl font-bold text-gray-900 mb-2">Sign in</h2>
      <p className="text-gray-600 mb-6 text-sm">Enter your UID to sign in.</p>
      {error && (
        <div className="bg-red-50 text-red-600 p-3 rounded-lg text-sm mb-4">
          {error}
        </div>
      )}
      <form onSubmit={handleSubmit} className="space-y-4">
        {/* UID input */}
        <div>
          <label className="block text-gray-900 font-medium mb-1 text-sm">
            UID <span className="text-red-500">*</span>
          </label>
          <input
            type="text"
            value={registrationNumber}
            onChange={(e) => setRegistrationNumber(e.target.value)}
            placeholder="e.g. 1230xxyyz"
            className="w-full px-3 py-2.5 bg-gray-100 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-gray-900"
            required
          />
        </div>

        {/* Password input */}
        <div>
          <label className="block text-gray-900 font-medium mb-1 text-sm">
            Password <span className="text-red-500">*</span>
          </label>
          <input
            type={showPassword ? "text" : "password"}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="Enter password"
            className="w-full px-3 py-2.5 bg-gray-100 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-gray-900"
            required
          />
        </div>

        {/* Submit button */}
        <button
          type="submit"
          disabled={isLoading}
          className={`w-full py-2.5 rounded-lg font-semibold transition-colors text-sm ${
            isLoading
              ? "bg-gray-400 text-white cursor-not-allowed"
              : "bg-gray-900 text-white hover:bg-gray-800 cursor-pointer"
          }`}
        >
          {isLoading ? "Signing in..." : "Sign in"}
        </button>
      </form>
    </div>
  );
};

export default AdminWhitePanel;
