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

module.exports = router;
