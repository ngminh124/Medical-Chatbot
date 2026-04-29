import { useEffect, useState } from "react";
import { adminAPI } from "../../api/admin";

export default function ConversationsPage() {
  const [threads, setThreads] = useState([]);
  const [selectedThread, setSelectedThread] = useState(null);
  const [messages, setMessages] = useState([]);

  useEffect(() => {
    adminAPI.listConversations({ page: 1, page_size: 50 }).then((res) => {
      setThreads(res.data.items || []);
    });
  }, []);

  const openThread = async (thread) => {
    setSelectedThread(thread);
    const res = await adminAPI.getConversationMessages(thread.id);
    setMessages(res.data.items || []);
  };

  return (
    <div className="grid gap-4 lg:grid-cols-[380px,1fr]">
      <div className="rounded-2xl border border-slate-800 bg-slate-900/80 p-3">
        <h3 className="mb-2 text-sm font-medium text-slate-200">Conversation threads</h3>
        <div className="max-h-[70vh] space-y-2 overflow-auto">
          {threads.map((thread) => (
            <button
              key={thread.id}
              onClick={() => openThread(thread)}
              className="w-full rounded-xl border border-slate-800 bg-slate-950/70 p-3 text-left hover:border-primary-500/40"
            >
              <p className="text-sm font-medium text-slate-100">{thread.title}</p>
              <p className="mt-1 text-xs text-slate-400">{thread.user_email}</p>
              <p className="mt-1 text-xs text-slate-500">
                {thread.message_count} msgs · {new Date(thread.updated_at).toLocaleString()}
              </p>
            </button>
          ))}
        </div>
      </div>

      <div className="rounded-2xl border border-slate-800 bg-slate-900/80 p-4">
        <h3 className="mb-3 text-sm font-medium text-slate-200">
          {selectedThread ? `Messages: ${selectedThread.title}` : "Select a thread"}
        </h3>

        <div className="max-h-[70vh] space-y-4 overflow-auto">
          {messages.map((item, idx) => (
            <div key={idx} className="rounded-xl border border-slate-800 bg-slate-950/50 p-3">
              <p className="text-xs font-medium uppercase tracking-wide text-slate-400">Question</p>
              <p className="mt-1 text-sm text-slate-200">{item.question}</p>

              <p className="mt-3 text-xs font-medium uppercase tracking-wide text-slate-400">Answer</p>
              <p className="mt-1 text-sm text-slate-300">{item.answer}</p>

              <div className="mt-2 flex gap-4 text-xs text-slate-500">
                <span>Latency: {item.latency ? `${item.latency.toFixed(2)}s` : "-"}</span>
                <span>{item.timestamp ? new Date(item.timestamp).toLocaleString() : "-"}</span>
              </div>
            </div>
          ))}
          {selectedThread && !messages.length && (
            <p className="text-sm text-slate-500">No Q/A pairs found.</p>
          )}
        </div>
      </div>
    </div>
  );
}
