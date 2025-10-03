const { createLogger, format, transports } = require("winston");
const { combine, timestamp, printf } = format;

const logFormat = printf(({ level, message, timestamp }) => {
  return `${timestamp} [${level.toUpperCase()}]: ${message}`;
});

const logger = createLogger({
  level: "info",
  format: combine(timestamp(), logFormat),
  transports: [
    new transports.Console(), // logs in terminal
    new transports.File({ filename: "logs/app.log" }) // logs in file
  ],
});

module.exports = logger;
