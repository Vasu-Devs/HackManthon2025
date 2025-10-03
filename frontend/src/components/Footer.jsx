const Footer = () => {
  return (
    <footer className="bg-gray-50 py-8">
      {/* Navigation Links */}
      <div className="max-w-6xl mx-auto px-8">
        <nav className="flex justify-center space-x-16 mb-8  gap-30">
          <a
            href="#about"
            className="text-gray-700 hover:text-gray-900 underline text-sm"
          >
            About
          </a>
          <a
            href="#team"
            className="text-gray-700 hover:text-gray-900 underline text-sm"
          >
            Our Team
          </a>
          <a
            href="#contact"
            className="text-gray-700 hover:text-gray-900 underline text-sm"
          >
            Contact
          </a>
          <a
            href="#top"
            className="text-gray-700 hover:text-gray-900 underline text-sm"
          >
            Go To Top
          </a>
        </nav>

        {/* Divider Line */}
        <div className="border-t border-gray-300 mb-8"></div>

        {/* Brand and Copyright */}
        <div className="text-center space-y-2">
          <div className="text-2xl font-medium text-gray-900">PolySee :)</div>
          <div className="text-xs text-gray-600">
            ©2025 PolySee. All Rights Reserved
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
