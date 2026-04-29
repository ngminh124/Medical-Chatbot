import { useCallback, useEffect, useState } from "react";
import { chatAPI } from "../api/chat";
import { useAuth } from "../contexts/AuthContext";
import { useTheme } from "../contexts/ThemeContext";
import {
  ChevronDown,
  ChevronUp,
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

const PAGE_LIMIT = 50;

export default function Sidebar({
  activeThreadId,
  onSelectThread,
  onNewChat,
  collapsed,
  onToggle,
  mobileOpen = false,
  onCloseMobile,
}) {
  const { user, logout } = useAuth();
  const { isDark, toggle: toggleTheme } = useTheme();

  const [threads, setThreads] = useState([]);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [threadsExpanded, setThreadsExpanded] = useState(true);
  const [skip, setSkip] = useState(0);
  const [hasMore, setHasMore] = useState(true);

  const fetchThreads = useCallback(async (nextSkip = 0, append = false) => {
    if (append) setLoadingMore(true);
    else setLoading(true);

    try {
      const res = await chatAPI.listThreads(nextSkip, PAGE_LIMIT);
      const items = Array.isArray(res?.data) ? res.data : [];

      setThreads((prev) => (append ? [...prev, ...items] : items));
      setSkip(nextSkip + items.length);
      setHasMore(items.length === PAGE_LIMIT);
    } catch {
      /* silent */
    } finally {
      setLoading(false);
      setLoadingMore(false);
    }
  }, []);

  useEffect(() => {
    fetchThreads(0, false);
  }, [fetchThreads]);

  const handleSelectThread = useCallback((id) => {
    onSelectThread(id);
    onCloseMobile?.();
  }, [onSelectThread, onCloseMobile]);

  const handleNewChat = useCallback(() => {
    onNewChat();
    onCloseMobile?.();
  }, [onNewChat, onCloseMobile]);

  const handleDelete = useCallback(async (e, threadId) => {
    e.stopPropagation();
    try {
      await chatAPI.deleteThread(threadId);
      setThreads((prev) => prev.filter((t) => t.id !== threadId));
      if (activeThreadId === threadId) {
        onNewChat();
      }
    } catch {
      /* silent */
    }
  }, [activeThreadId, onNewChat]);

  const handleThreadScroll = useCallback(async (e) => {
    const node = e.currentTarget;
    if (!node || loading || loadingMore || !hasMore) return;

    const threshold = 100;
    const reachedBottom = node.scrollTop + node.clientHeight >= node.scrollHeight - threshold;
    if (!reachedBottom) return;

    await fetchThreads(skip, true);
  }, [fetchThreads, hasMore, loading, loadingMore, skip]);

  return (
    <>
      {mobileOpen && (
        <button
          className="fixed inset-0 z-30 bg-black/40 lg:hidden"
          onClick={onCloseMobile}
          aria-label="Đóng sidebar"
        />
      )}

      <aside
        className={`fixed inset-y-0 left-0 z-40 flex w-80 max-w-[90vw] transform flex-col border-r border-gray-200 bg-white transition-transform duration-200 dark:border-gray-700 dark:bg-gray-900 sm:w-96 lg:static lg:z-auto lg:h-full lg:max-w-none lg:translate-x-0 ${
          mobileOpen ? "translate-x-0" : "-translate-x-full"
        } ${collapsed ? "lg:w-24" : ""}`}
      >
        <div className="flex items-center justify-between border-b border-gray-100 px-4 py-3 dark:border-gray-800">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary-600">
              <Heart className="h-6 w-6 text-white" />
            </div>
            {!collapsed && (
              <span className="text-base font-semibold text-gray-900 dark:text-gray-100">
                Minqes
              </span>
            )}
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={onCloseMobile}
              className="flex h-10 w-10 items-center justify-center rounded-lg text-gray-400 transition-colors hover:bg-gray-100 hover:text-gray-600 dark:text-gray-500 dark:hover:bg-gray-800 dark:hover:text-gray-300 lg:hidden"
              title="Đóng sidebar"
            >
              <PanelLeftClose className="h-6 w-6" />
            </button>
            <button
              onClick={toggleTheme}
              className="flex h-10 w-10 items-center justify-center rounded-lg text-gray-400 transition-colors hover:bg-gray-100 hover:text-gray-600 dark:text-gray-500 dark:hover:bg-gray-800 dark:hover:text-gray-300"
              title={isDark ? "Chế độ sáng" : "Chế độ tối"}
            >
              {isDark ? <Sun className="h-6 w-6 text-amber-400" /> : <Moon className="h-6 w-6" />}
            </button>
            <button
              onClick={onToggle}
              className="hidden h-10 w-10 items-center justify-center rounded-lg text-gray-400 transition-colors hover:bg-gray-100 hover:text-gray-600 dark:text-gray-500 dark:hover:bg-gray-800 dark:hover:text-gray-300 lg:flex"
              title="Thu gọn sidebar"
            >
              <PanelLeftClose className="h-6 w-6" />
            </button>
          </div>
        </div>

        {!collapsed && (
          <>
            <div className="p-4">
              <button
                onClick={handleNewChat}
                className="flex w-full items-center gap-3 rounded-lg border border-gray-200 px-4 py-3 text-sm font-medium text-gray-700 transition-colors hover:bg-gray-50 dark:border-gray-700 dark:text-gray-300 dark:hover:bg-gray-800"
              >
                <span className="flex h-6 w-6 items-center justify-center">
                  <MessageSquarePlus className="h-6 w-6" />
                </span>
                Cuộc trò chuyện mới
              </button>
            </div>

            <div
              className="flex-1 min-h-0 overflow-y-auto px-4 pb-4"
              style={{ height: "100%", maxHeight: "100vh", overflowY: "auto" }}
              onScroll={handleThreadScroll}
            >
              <button
                onClick={() => setThreadsExpanded((prev) => !prev)}
                className="group mb-2 flex w-full items-center justify-between rounded-lg px-2 py-2 text-left text-sm font-semibold text-gray-700 transition-colors hover:bg-gray-50 dark:text-gray-200 dark:hover:bg-gray-800"
              >
                <span>Các cuộc trò chuyện của tôi</span>
                <span className="flex h-6 w-6 items-center justify-center text-gray-400 opacity-0 transition-opacity group-hover:opacity-100 dark:text-gray-500">
                  {threadsExpanded ? <ChevronUp className="h-6 w-6" /> : <ChevronDown className="h-6 w-6" />}
                </span>
              </button>

              <div
                className={`overflow-hidden transition-all duration-300 ease-in-out ${
                  threadsExpanded ? "max-h-[100vh] opacity-100" : "max-h-0 opacity-0"
                }`}
              >
                {loading ? (
                  <div className="space-y-2">
                    {[...Array(5)].map((_, i) => (
                      <div key={i} className="skeleton h-12 rounded-lg" />
                    ))}
                  </div>
                ) : threads.length === 0 ? (
                  <div className="mt-8 text-center">
                    <MessagesSquare className="mx-auto h-10 w-10 text-gray-300 dark:text-gray-600" />
                    <p className="mt-2 text-sm text-gray-400 dark:text-gray-500">
                      Chưa có cuộc trò chuyện nào
                    </p>
                  </div>
                ) : (
                  <div className="space-y-2">
                    {threads.map((thread) => (
                      <button
                        key={thread.id}
                        onClick={() => handleSelectThread(thread.id)}
                        className={`group flex w-full items-start gap-3 rounded-lg px-3 py-3 text-left text-sm transition-colors ${
                          activeThreadId === thread.id
                            ? "bg-primary-50 text-primary-700 dark:bg-primary-900/30 dark:text-primary-300"
                            : "text-gray-700 hover:bg-gray-50 dark:text-gray-300 dark:hover:bg-gray-800"
                        }`}
                      >
                        <span className="mt-0.5 flex h-6 w-6 flex-shrink-0 items-center justify-center opacity-60">
                          <MessagesSquare className="h-6 w-6" />
                        </span>
                        <div className="min-w-0 flex-1">
                          <p className="truncate text-sm font-medium">{thread.title}</p>
                          {thread.last_message && (
                            <p className="mt-1 truncate text-sm text-gray-400 dark:text-gray-500">
                              {thread.last_message}
                            </p>
                          )}
                        </div>
                        <button
                          onClick={(e) => handleDelete(e, thread.id)}
                          className="mt-0.5 flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-lg text-gray-300 opacity-0 transition-opacity hover:bg-red-50 hover:text-red-500 group-hover:opacity-100 dark:text-gray-600 dark:hover:bg-red-900/20"
                          title="Xóa"
                        >
                          <Trash2 className="h-6 w-6" />
                        </button>
                      </button>
                    ))}

                    {loadingMore && (
                      <div className="py-2 text-center text-sm text-gray-400 dark:text-gray-500">
                        Đang tải thêm...
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>

            <div className="border-t border-gray-100 p-4 dark:border-gray-800">
              <div className="flex items-center gap-3 rounded-lg px-2 py-2">
                <div className="flex h-10 w-10 items-center justify-center rounded-full bg-primary-100 text-primary-600 dark:bg-primary-900/30 dark:text-primary-400">
                  <User className="h-6 w-6" />
                </div>
                <div className="min-w-0 flex-1">
                  <p className="truncate text-sm font-medium text-gray-900 dark:text-gray-100">
                    {user?.name}
                  </p>
                  <p className="truncate text-sm text-gray-400 dark:text-gray-500">{user?.email}</p>
                </div>
                <button
                  onClick={logout}
                  className="flex h-10 w-10 items-center justify-center rounded-lg text-gray-400 hover:bg-red-50 hover:text-red-500 dark:text-gray-500 dark:hover:bg-red-900/20"
                  title="Đăng xuất"
                >
                  <LogOut className="h-6 w-6" />
                </button>
              </div>
            </div>
          </>
        )}
      </aside>
    </>
  );
}


