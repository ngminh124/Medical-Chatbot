import client from "./client";

const API_BASE_URL = import.meta.env.VITE_API_URL || "";

function buildApiUrl(path) {
  if (!API_BASE_URL) return path;
  if (/^https?:\/\//i.test(API_BASE_URL)) {
    return `${API_BASE_URL.replace(/\/$/, "")}${path}`;
  }
  return `${API_BASE_URL}${path}`;
}

function parseSSEEvent(rawEvent) {
  let event = "message";
  const dataLines = [];

  rawEvent.split(/\r?\n/).forEach((line) => {
    if (line.startsWith("event:")) {
      event = line.slice(6).trim() || "message";
      return;
    }
    if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trimStart());
    }
  });

  if (!dataLines.length) return null;
  return { event, data: dataLines.join("\n") };
}

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
  ask: (threadId, content, options = {}, requestConfig = {}) =>
    client.post(
      `/v1/chat/threads/${threadId}/ask`,
      {
        content,
        ...(typeof options.web_search_enabled === "boolean"
          ? { web_search_enabled: options.web_search_enabled }
          : {}),
      },
      { timeout: 300_000, ...requestConfig },
    ),

  /**
   * POST /v1/chat/threads/:id/ask-stream
   * Reads Server-Sent Events and emits chunks via callbacks.
   */
  askStream: async (threadId, content, options = {}, streamConfig = {}) => {
    const {
      signal,
      onChunk,
      onOpen,
      onEvent,
      onFinal,
    } = streamConfig;

    const token = sessionStorage.getItem("access_token");
    const response = await fetch(
      buildApiUrl(`/v1/chat/threads/${threadId}/ask-stream`),
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "text/event-stream",
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify({
          content,
          ...(typeof options.web_search_enabled === "boolean"
            ? { web_search_enabled: options.web_search_enabled }
            : {}),
        }),
        signal,
      },
    );

    if (!response.ok) {
      const err = new Error("Streaming request failed");
      err.status = response.status;
      throw err;
    }

    if (!response.body) {
      const err = new Error("Streaming body is unavailable");
      err.status = 500;
      throw err;
    }

    onOpen?.();

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";
    let assistantText = "";
    let finalUserMessage = null;
    let finalAssistantMessage = null;

    const processRawEvent = (rawEvent) => {
      const parsed = parseSSEEvent(rawEvent);
      if (!parsed) return;

      const { event, data } = parsed;
      let payload;
      try {
        payload = JSON.parse(data);
      } catch {
        payload = data;
      }

      onEvent?.({ event, payload });

      if (typeof payload === "string") {
        if (payload) {
          assistantText += payload;
          onChunk?.(payload, assistantText, null);
        }
        return;
      }

      if (payload?.error) {
        const err = new Error(payload.error || "Streaming error");
        err.status = 500;
        throw err;
      }

      const chunk = payload?.chunk ?? payload?.delta ?? payload?.token ?? payload?.content ?? "";
      if (chunk) {
        assistantText += chunk;
        onChunk?.(chunk, assistantText, payload);
      }

      if (payload?.user_message) {
        finalUserMessage = payload.user_message;
      }
      if (payload?.assistant_message) {
        finalAssistantMessage = payload.assistant_message;
      }

      if (event === "done" || payload?.done === true) {
        onFinal?.({
          user_message: finalUserMessage,
          assistant_message: finalAssistantMessage,
          assistant_text: assistantText,
          payload,
        });
      }
    };

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      let boundary = buffer.indexOf("\n\n");
      while (boundary !== -1) {
        const rawEvent = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        processRawEvent(rawEvent);
        boundary = buffer.indexOf("\n\n");
      }
    }

    const tail = decoder.decode();
    if (tail) {
      buffer += tail;
    }

    if (buffer.trim()) {
      processRawEvent(buffer.trim());
    }

    return {
      user_message: finalUserMessage,
      assistant_message: finalAssistantMessage,
      assistant_text: assistantText,
    };
  },

  // ── Feedbacks ─────────────────────────────────────────────
  /** POST /v1/chat/messages/:id/feedback */
  createFeedback: (messageId, rating, comment) =>
    client.post(`/v1/chat/messages/${messageId}/feedback`, {
      rating,
      comment,
    }),
};
