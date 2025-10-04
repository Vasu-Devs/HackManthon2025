const express = require("express");
const router = express.Router();
const auth = require("../middleware/auth");
const User = require("../models/User");
const axios = require("axios");

// get recent chats for logged-in user
router.get("/user/recent-chats", auth(["user", "admin"]), async (req, res) => {
  const regNo = req.user.regNo;
  const user = await User.findOne({ regNo }, { recentChats: 1, _id: 0 });
  res.json({ recentChats: user?.recentChats || [] });
});

// in routes/api.js
router.get("/user/:regNo", auth(["user", "admin"]), async (req, res) => {
  const regNo = req.params.regNo;
  const user = await User.findOne({ regNo }, { email: 1, regNo: 1, _id: 0 });
  if (!user) return res.status(404).json({ error: "User not found" });
  res.json(user);
});


// gateway: forward chat to FastAPI and save to recentChats
router.post("/chat", auth(["user", "admin"]), async (req, res) => {
  const { query } = req.body;
  const regNo = req.user.regNo;
  try {
    const resp = await axios.post(
      `${process.env.FASTAPI_URL}/v1/chat`,
      { query },
      {
        headers: { Authorization: req.headers.authorization },
      }
    );

    const {
      answer,
      success = true,
      responseTimeMs = 0,
      language = "en",
      category = "general",
    } = resp.data;

    // push to user's recentChats
    await User.updateOne(
      { regNo },
      {
        $push: {
          recentChats: {
            query,
            response: answer,
            success,
            responseTimeMs,
            language,
            category,
          },
        },
      }
    );

    res.json({ answer, success, responseTimeMs, language, category });
  } catch (err) {
    console.error(err?.response?.data || err.message);
    res.status(500).json({ error: "ai-failed" });
  }
});

// get user email by regNo
router.get("/user/:regNo", auth(["user", "admin"]), async (req, res) => {
  try {
    const regNo = req.params.regNo;

    // Only fetch email + regNo, don’t expose password hash or other sensitive data
    const user = await User.findOne(
      { regNo },
      { email: 1, regNo: 1, role: 1, _id: 0 }
    );

    if (!user) {
      return res.status(404).json({ error: "User not found" });
    }

    res.json(user);
  } catch (err) {
    console.error("❌ Error fetching user:", err.message);
    res.status(500).json({ error: "Internal server error" });
  }
});


module.exports = router;
