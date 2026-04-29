import client from "./client";

export const adminAPI = {
  getOverview: () => client.get("/v1/admin/overview"),

  listUsers: (params) => client.get("/v1/admin/users", { params }),

  listConversations: (params) => client.get("/v1/admin/conversations", { params }),
  getConversationMessages: (threadId) =>
    client.get(`/v1/admin/conversations/${threadId}/messages`),

  queryMetricsRange: ({ query, start, end, step = "5s" }) =>
    client.get("/v1/admin/metrics/range", {
      params: { query, start, end, step, _ts: Date.now() },
      headers: {
        "Cache-Control": "no-cache",
        Pragma: "no-cache",
      },
    }),

  getLogs: ({ level = "", keyword = "", limit = 200 } = {}) =>
    client.get("/v1/admin/logs", {
      params: { level, keyword, limit },
    }),

  getTraces: ({ service = "rag_pipeline", limit = 20 } = {}) =>
    client.get("/v1/admin/traces", {
      params: { service, limit },
    }),

  getSettings: () => client.get("/v1/admin/settings"),
  updateSettings: (payload) => client.put("/v1/admin/settings", payload),
};
