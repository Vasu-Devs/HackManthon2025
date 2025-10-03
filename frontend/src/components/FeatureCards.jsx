const FeatureCards = () => {
  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-4xl mx-auto p-6">
      {/* Card 1 - 24/7 active */}
      <div className="bg-white rounded-lg p-6 shadow-sm border border-gray-100">
        <div className="flex flex-col h-full">
          <div className="flex-1">
            <h3 className="text-lg font-semibold text-gray-900 mb-1">
              24/7 active
            </h3>
            <p className="text-sm text-gray-500">Latest AI Support</p>
          </div>
          <div className="mt-8 flex justify-end">
            <div className="bg-black rounded-lg w-8 h-8 flex items-center justify-center">
              <div className="w-2 h-2 bg-white rounded-full"></div>
            </div>
          </div>
        </div>
      </div>

      {/* Card 2 - 5 Language */}
      <div className="bg-white rounded-lg p-6 shadow-sm border border-gray-100">
        <div className="flex flex-col h-full">
          <div className="flex-1">
            <h3 className="text-lg font-semibold text-gray-900 mb-1">
              50+ Languages
            </h3>
            <p className="text-sm text-gray-500">
              Multi Linguistic Interaction
            </p>
          </div>
          <div className="mt-8 flex justify-end">
            <div className="bg-black rounded-lg w-8 h-8 flex items-center justify-center relative">
              <div className="w-1.5 h-1.5 bg-white rounded-full absolute top-2 left-2"></div>
              <div className="w-1.5 h-1.5 bg-white rounded-full absolute bottom-2 right-2"></div>
            </div>
          </div>
        </div>
      </div>

      {/* Card 3 - Secure & Reliable */}
      <div className="bg-white rounded-lg p-6 shadow-sm border border-gray-100">
        <div className="flex flex-col h-full">
          <div className="flex-1">
            <h3 className="text-lg font-semibold text-gray-900 mb-1">
              Secure & Reliable
            </h3>
            <p className="text-sm text-gray-500">Latest AI Support</p>
          </div>
          <div className="mt-8 flex justify-end">
            <div className="bg-black rounded-lg w-8 h-8 flex items-center justify-center relative">
              <div className="w-1.5 h-1.5 bg-white rounded-full absolute top-1.5 left-2"></div>
              <div className="w-1.5 h-1.5 bg-white rounded-full absolute top-1.5 right-2"></div>
              <div className="w-1.5 h-1.5 bg-white rounded-full absolute bottom-1.5 left-1/2 transform -translate-x-1/2"></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default FeatureCards;
