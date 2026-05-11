import axios from "axios";

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "";

console.log("API_BASE_URL =", API_BASE_URL);

export const api = axios.create({
  baseURL: API_BASE_URL,
  withCredentials: true,
  headers: { "Content-Type": "application/json" },
  timeout: 300_000,
});

export const buildApiUrl = (path: string) => {
  if (!API_BASE_URL) return path;
  if (/^https?:\/\//i.test(API_BASE_URL)) {
    return `${API_BASE_URL.replace(/\/$/, "")}${path}`;
  }
  return `${API_BASE_URL}${path}`;
};
