import { useEffect, useState } from "react";
import { chatAPI } from "../api/chat";
import { useAuth } from "../contexts/AuthContext";
import { useTheme } from "../contexts/ThemeContext";
import {
  Heart,
  LogOut,
  MessageSquarePlus,
  MessagesSquare,
  Moon,
  PanelLeftClose,
  Sun,
  Trash2,
  User,
} from "lucide-react";

export default function Sidebar({
  activeThreadId,
  onSelectThread,
  onNewChat,
  collapsed,
  onToggle,
}) {
  const { user, logout } = useAuth();
  const { isDark, toggle: toggleTheme } = useTheme();
  const [threads, setThreads] = useState([]);
  const [loading, setLoading] = useState(true);

  const fetchThreads = async () => {
    try {
      const res = await chatAPI.listThreads();
      setThreads(res.data);
    } catch {
      /* silent */
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchThreads(); }, []);
  useEffect(() => { if (activeThreadId) fetchThreads(); }, [activeThreadId]);

  const handleDelete = async (e, threadId) => {
    e.stopPropagation();
    if (!confirm("Bạn có chắc muốn xóa cuộc trò chuyện này?")) return;
    try {
      await chatAPI.deleteThread(threadId);
      setThreads((prev) => prev.filter((t) => t.id !== threadId));
      if (activeThreadId === threadId) onNewChat();
    } catch {
      /* silent */
    }
  };

  /* ── Collapsed state ──────────────────────────────────────── */
  if (collapsed) {
    return (
      <div className="flex h-full w-16 flex-col items-center border-r border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 py-4 gap-3">
        <button
          onClick={onToggle}
          className="rounded-lg p-2 text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800"
          title="Mở sidebar"
        >
          <MessagesSquare className="h-5 w-5" />
        </button>
        <button
          onClick={onNewChat}
          className="rounded-lg bg-primary-600 p-2 text-white hover:bg-primary-700"
          title="Cuộc trò chuyện mới"
        >
          <MessageSquarePlus className="h-5 w-5" />
        </button>
        <div className="flex-1" />
        <button
          onClick={toggleTheme}
          className="rounded-lg p-2 text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800"
          title={isDark ? "Chế độ sáng" : "Chế độ tối"}
        >
          {isDark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
        </button>
      </div>
    );
  }

  /* ── Expanded state ───────────────────────────────────────── */
  return (
    <div className="flex h-full w-72 flex-col border-r border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-gray-100 dark:border-gray-800 px-4 py-3">
        <div className="flex items-center gap-2">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary-600">
            <Heart className="h-4 w-4 text-white" />
          </div>
          <span className="text-sm font-semibold text-gray-900 dark:text-gray-100">
            Medical Chat
          </span>
        </div>
        <div className="flex items-center gap-1">
          {/* Theme toggle */}
          <button
            onClick={toggleTheme}
            className="rounded-lg p-1.5 text-gray-400 dark:text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-800 hover:text-gray-600 dark:hover:text-gray-300 transition-colors"
            title={isDark ? "Chế độ sáng" : "Chế độ tối"}
          >
            {isDark
              ? <Sun className="h-4 w-4 text-amber-400" />
              : <Moon className="h-4 w-4" />
            }
          </button>
          {/* Collapse */}
          <button
            onClick={onToggle}
            className="rounded-lg p-1.5 text-gray-400 dark:text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-800 hover:text-gray-600 dark:hover:text-gray-300 transition-colors"
            title="Thu gọn sidebar"
          >
            <PanelLeftClose className="h-4 w-4" />
          </button>
        </div>
      </div>

      {/* New Chat Button */}
      <div className="p-3">
        <button
          onClick={onNewChat}
          className="flex w-full items-center gap-2 rounded-lg border border-gray-200 dark:border-gray-700 px-3 py-2.5 text-sm font-medium text-gray-700 dark:text-gray-300 transition-colors hover:bg-gray-50 dark:hover:bg-gray-800"
        >
          <MessageSquarePlus className="h-4 w-4" />
          Cuộc trò chuyện mới
        </button>
      </div>

      {/* Thread List */}
      <div className="flex-1 overflow-y-auto px-3 pb-3">
        {loading ? (
          <div className="space-y-2">
            {[...Array(5)].map((_, i) => (
              <div key={i} className="skeleton h-12 rounded-lg" />
            ))}
          </div>
        ) : threads.length === 0 ? (
          <div className="mt-8 text-center">
            <MessagesSquare className="mx-auto h-8 w-8 text-gray-300 dark:text-gray-600" />
            <p className="mt-2 text-xs text-gray-400 dark:text-gray-500">
              Chưa có cuộc trò chuyện nào
            </p>
          </div>
        ) : (
          <div className="space-y-1">
            {threads.map((thread) => (
              <button
                key={thread.id}
                onClick={() => onSelectThread(thread.id)}
                className={`group flex w-full items-start gap-2 rounded-lg px-3 py-2.5 text-left text-sm transition-colors ${
                  activeThreadId === thread.id
                    ? "bg-primary-50 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300"
                    : "text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-800"
                }`}
              >
                <MessagesSquare className="mt-0.5 h-4 w-4 flex-shrink-0 opacity-50" />
                <div className="min-w-0 flex-1">
                  <p className="truncate font-medium">{thread.title}</p>
                  {thread.last_message && (
                    <p className="mt-0.5 truncate text-xs text-gray-400 dark:text-gray-500">
                      {thread.last_message}
                    </p>
                  )}
                </div>
                <button
                  onClick={(e) => handleDelete(e, thread.id)}
                  className="mt-0.5 flex-shrink-0 rounded p-1 text-gray-300 dark:text-gray-600 opacity-0 transition-opacity hover:bg-red-50 dark:hover:bg-red-900/20 hover:text-red-500 group-hover:opacity-100"
                  title="Xóa"
                >
                  <Trash2 className="h-3.5 w-3.5" />
                </button>
              </button>
            ))}
          </div>
        )}
      </div>

      {/* User Info */}
      <div className="border-t border-gray-100 dark:border-gray-800 p-3">
        <div className="flex items-center gap-3 rounded-lg px-3 py-2">
          <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary-100 dark:bg-primary-900/30 text-primary-600 dark:text-primary-400">
            <User className="h-4 w-4" />
          </div>
          <div className="min-w-0 flex-1">
            <p className="truncate text-sm font-medium text-gray-900 dark:text-gray-100">
              {user?.name}
            </p>
            <p className="truncate text-xs text-gray-400 dark:text-gray-500">{user?.email}</p>
          </div>
          <button
            onClick={logout}
            className="rounded-lg p-1.5 text-gray-400 dark:text-gray-500 hover:bg-red-50 dark:hover:bg-red-900/20 hover:text-red-500"
            title="Đăng xuất"
          >
            <LogOut className="h-4 w-4" />
          </button>
        </div>
      </div>
    </div>
  );
}


