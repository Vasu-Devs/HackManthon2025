const jwt = require("jsonwebtoken");
const JWT_SECRET = process.env.JWT_SECRET;

module.exports = function (requiredRoles = []) {
  return (req, res, next) => {
    const auth = req.headers.authorization;
    if (!auth) return res.status(401).json({ error: "no token" });
    const token = auth.split(" ")[1];
    try {
      const payload = jwt.verify(token, JWT_SECRET);
      if (requiredRoles.length && !requiredRoles.includes(payload.role)) {
        return res.status(403).json({ error: "forbidden" });
      }
      req.user = payload; // { regNo, role, iat, exp }
      next();
    } catch (err) {
      return res.status(401).json({ error: "invalid token" });
    }
  };
};
