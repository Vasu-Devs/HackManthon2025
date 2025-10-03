const express = require("express");
const router = express.Router();
const User = require("../models/User");
const bcrypt = require("bcryptjs");
const jwt = require("jsonwebtoken");

// Log 
const logger = require("../utils/logger");

const JWT_SECRET = process.env.JWT_SECRET;

// Health check route
router.get("/health", (req, res) => {
  logger.info("Health check called"); // log health checks
  res.json({
    status: "Auth service is running",
    timestamp: new Date().toISOString(),
  });
});

// REGISTER: expects { regNo, email, password, role }
router.post("/register", async (req, res) => {
  const { regNo, email, password, role } = req.body;

  // Validate required fields
  if (!regNo || !email || !password || !role) {
    logger.warn(`Failed registration attempt - Missing fields`);
    return res
      .status(400)
      .json({ error: "Missing required fields: regNo, email, password, role" });
  }

  // Validate role
  if (!["user", "admin"].includes(role)) {
    logger.warn(`Invalid role '${role}' attempted for regNo: ${regNo}`);
    return res.status(400).json({ error: "Role must be 'user' or 'admin'" });
  }

  try {
    // Check if user already exists (by regNo or email)
    const existingUser = await User.findOne({ $or: [{ regNo }, { email }] });
    if (existingUser) {
      logger.warn(`Registration failed - User already exists: ${regNo} or ${email}`);
      return res
        .status(409)
        .json({ error: "User with this registration number or email already exists" });
    }

    // Hash password
    const saltRounds = 10;
    const passwordHash = await bcrypt.hash(password, saltRounds);

    // Create new user
    const newUser = new User({
      regNo,
      email,
      passwordHash,
      role,
    });

    await newUser.save();

    logger.info(`New user registered: ${regNo}, role: ${role}`);

    res.status(201).json({
      message: "User registered successfully",
      regNo: newUser.regNo,
      role: newUser.role,
    });
  } catch (error) {
    logger.error(`Registration error for ${regNo}: ${error.message}`);
    res.status(500).json({ error: "Internal server error" });
  }
});

// LOGIN: expects { regNo, password }
router.post("/login", async (req, res) => {
  const { regNo, password } = req.body;
  if (!regNo || !password) {
    logger.warn("Login attempt with missing fields");
    return res.status(400).json({ error: "missing fields" });
  }

  const user = await User.findOne({ regNo });
  if (!user) {
    logger.warn(`Invalid login attempt - User not found: ${regNo}`);
    return res.status(401).json({ error: "invalid credentials" });
  }

  const match = await bcrypt.compare(password, user.passwordHash);
  if (!match) {
    logger.warn(`Invalid login attempt - Wrong password for ${regNo}`);
    return res.status(401).json({ error: "invalid credentials" });
  }

  const token = jwt.sign({ regNo: user.regNo, role: user.role }, JWT_SECRET, {
    expiresIn: "24h",
  });

  logger.info(`User logged in: ${regNo}, role: ${user.role}`);

  return res.json({ token, regNo: user.regNo, role: user.role });
});

// VERIFY token: return payload
router.get("/me", (req, res) => {
  const auth = req.headers.authorization;
  if (!auth) {
    logger.warn("Token verification failed - Missing Authorization header");
    return res.status(401).end();
  }
  const token = auth.split(" ")[1];
  try {
    const payload = jwt.verify(token, JWT_SECRET);
    logger.info(`Token verified for user: ${payload.regNo}`);
    res.json(payload);
  } catch (e) {
    logger.warn("Token verification failed - Invalid/Expired token");
    res.status(401).end();
  }
});

module.exports = router;
