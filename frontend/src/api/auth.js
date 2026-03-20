import client from "./client";

export const authAPI = {
  /** POST /v1/auth/register */
  register: (data) => client.post("/v1/auth/register", data),

  /** POST /v1/auth/login */
  login: (data) => client.post("/v1/auth/login", data),

  /** GET /v1/auth/me */
  getMe: () => client.get("/v1/auth/me"),
};
