import client from "./client";

export const chatAPI = {
  // ── Threads ───────────────────────────────────────────────
  /** POST /v1/chat/threads */
  createThread: (title) =>
    client.post("/v1/chat/threads", { title: title || "Cuộc trò chuyện mới" }),

  /** GET /v1/chat/threads */
  listThreads: (skip = 0, limit = 50) =>
    client.get("/v1/chat/threads", { params: { skip, limit } }),

  /** GET /v1/chat/threads/:id */
  getThread: (threadId) => client.get(`/v1/chat/threads/${threadId}`),

  /** DELETE /v1/chat/threads/:id */
  deleteThread: (threadId) => client.delete(`/v1/chat/threads/${threadId}`),

  // ── Messages ──────────────────────────────────────────────
  /** GET /v1/chat/threads/:id/messages */
  listMessages: (threadId, skip = 0, limit = 100) =>
    client.get(`/v1/chat/threads/${threadId}/messages`, {
      params: { skip, limit },
    }),

  /** POST /v1/chat/threads/:id/messages  (plain persist, no AI) */
  sendMessage: (threadId, content) =>
    client.post(`/v1/chat/threads/${threadId}/messages`, { content }),

  /**
   * POST /v1/chat/threads/:id/ask
   * Persist user message AND trigger the full RAG pipeline in one call.
   * Returns { user_message, assistant_message, citations, route }.
   * Uses a 300-second timeout to accommodate long LLM generation time.
   */
  ask: (threadId, content) =>
    client.post(
      `/v1/chat/threads/${threadId}/ask`,
      { content },
      { timeout: 300_000 },
    ),

  // ── Feedbacks ─────────────────────────────────────────────
  /** POST /v1/chat/messages/:id/feedback */
  createFeedback: (messageId, rating, comment) =>
    client.post(`/v1/chat/messages/${messageId}/feedback`, {
      rating,
      comment,
    }),
};
