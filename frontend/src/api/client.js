import axios from "axios";

const API_BASE_URL = import.meta.env.VITE_API_URL || "";

const client = axios.create({
  baseURL: API_BASE_URL,
  headers: { "Content-Type": "application/json" },
  timeout: 300_000,
});

// ── Request interceptor: attach JWT ─────────────────────────
client.interceptors.request.use((config) => {
  const token = sessionStorage.getItem("access_token");
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// ── Response interceptor: handle 401 ────────────────────────
client.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      sessionStorage.removeItem("access_token");
      sessionStorage.removeItem("user");
      sessionStorage.removeItem("stt_prompted_this_session");
      sessionStorage.removeItem("tts_prompted_this_session");

      // Cleanup any legacy persistent auth from old versions
      localStorage.removeItem("access_token");
      localStorage.removeItem("user");
      // Only redirect if not already on auth pages
      if (
        !window.location.pathname.includes("/login") &&
        !window.location.pathname.includes("/register")
      ) {
        window.location.href = "/login";
      }
    }
    return Promise.reject(error);
  }
);

export default client;
