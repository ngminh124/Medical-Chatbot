import { useState } from "react";
import SidebarPanel from "../components/SidebarPanel";
import ChatWindow from "../components/ChatWindow";
import { Menu } from "lucide-react";

export default function ChatPage() {
  const [activeThreadId, setActiveThreadId] = useState(null);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);

  return (
    <div className="flex h-screen overflow-hidden bg-gray-50 dark:bg-gray-900">
      <SidebarPanel
        activeThreadId={activeThreadId}
        onSelectThread={setActiveThreadId}
        onNewChat={() => setActiveThreadId(null)}
        collapsed={sidebarCollapsed}
        onToggle={() => setSidebarCollapsed(!sidebarCollapsed)}
        mobileOpen={mobileSidebarOpen}
        onCloseMobile={() => setMobileSidebarOpen(false)}
      />
      <main className="relative min-w-0 flex-1 overflow-hidden">
        <button
          onClick={() => setMobileSidebarOpen(true)}
          className="absolute left-3 top-3 z-20 flex h-10 w-10 items-center justify-center rounded-lg border border-gray-200 bg-white text-gray-600 shadow-sm hover:bg-gray-50 dark:border-gray-700 dark:bg-gray-800 dark:text-gray-300 dark:hover:bg-gray-700 lg:hidden"
          title="Mở lịch sử hội thoại"
        >
          <Menu className="h-6 w-6" />
        </button>
        <ChatWindow
          threadId={activeThreadId}
          onThreadCreated={setActiveThreadId}
        />
      </main>
    </div>
  );
}
