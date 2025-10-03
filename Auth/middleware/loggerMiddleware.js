const logger = require("../utils/logger");

const requestLogger = (req, res, next) => {
  logger.info(`${req.method} ${req.url} - ${req.ip}`);
  next();
};

module.exports = requestLogger;
