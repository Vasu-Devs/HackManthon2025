const mongoose = require("mongoose");
const Schema = mongoose.Schema;

const ChatSchema = new Schema({
  query: String,
  response: String,
  success: Boolean,
  responseTimeMs: Number,
  language: String,
  category: String,
  createdAt: { type: Date, default: Date.now },
});

const FileMetaSchema = new Schema({
  filename: String,
  uploadedAt: { type: Date, default: Date.now },
  uploaderRegNo: String,
});

const UserSchema = new Schema({
  regNo: { type: String, unique: true },
  email: { type: String, unique: true, required: true },
  passwordHash: String, // bcrypt hash expected
  role: { type: String, enum: ["user", "admin"], default: "user" },
  recentChats: [ChatSchema],
  uploadedFiles: [FileMetaSchema],
});

module.exports = mongoose.model("User", UserSchema);
