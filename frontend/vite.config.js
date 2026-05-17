import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const apiBaseUrl = env.VITE_API_BASE_URL;
  const proxy = apiBaseUrl
    ? {
        "/v1": {
          target: apiBaseUrl,
          changeOrigin: true,
          secure: apiBaseUrl.startsWith("https://"),
        },
      }
    : undefined;

  return {
    plugins: [react()],
    server: {
      port: 3000,
      ...(proxy ? { proxy } : {}),
    },
  };
});
