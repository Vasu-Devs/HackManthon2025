require("dotenv").config();
const express = require("express");
const mongoose = require("mongoose");
const cors = require("cors");

const authRoutes = require("./routes/auth");
const apiRoutes = require("./routes/api"); // protected user/admin 
const requestLogger = require("./middleware/loggerMiddleware"); //For logging

const app = express();
app.use(express.json());

// allow only the frontend origin
app.use(cors({ origin: process.env.FRONTEND_ORIGIN, credentials: true }));

mongoose
  .connect(process.env.MONGO_URI, {
    useNewUrlParser: true,
    useUnifiedTopology: true,
  })
  .then(() => console.log("Mongo connected"))
  .catch((err) => console.error("Mongo error", err));

app.use("/auth", authRoutes);
app.use("/api", apiRoutes);
app.use(requestLogger); //Log route middleware

const PORT = process.env.PORT || 4000;
app.listen(PORT, () => console.log("Express listening on", PORT));
