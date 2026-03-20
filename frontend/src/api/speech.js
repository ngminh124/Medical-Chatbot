import client from "./client";

export const speechAPI = {
  /** GET /v1/stt/health */
  sttHealth: () => client.get("/v1/stt/health", { timeout: 10_000 }),

  /** GET /v1/tts/health */
  ttsHealth: () => client.get("/v1/tts/health", { timeout: 10_000 }),

  /** POST /v1/stt/transcribe (multipart/form-data, field: file) */
  transcribe: (formData, config = {}) =>
    client.post("/v1/stt/transcribe", formData, {
      ...config,
      headers: {
        "Content-Type": "multipart/form-data",
        ...(config.headers || {}),
      },
      timeout: config.timeout ?? 120_000,
    }),

  /** POST /v1/tts/synthesize -> audio/mpeg */
  synthesize: (payload, config = {}) =>
    client.post("/v1/tts/synthesize", payload, {
      ...config,
      responseType: "blob",
      timeout: config.timeout ?? 120_000,
    }),
};
