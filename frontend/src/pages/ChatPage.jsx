import { useState } from "react";
import Sidebar from "../components/Sidebar";
import ChatWindow from "../components/ChatWindow";

export default function ChatPage() {
  const [activeThreadId, setActiveThreadId] = useState(null);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  return (
    <div className="flex h-screen bg-gray-50 dark:bg-gray-900">
      <Sidebar
        activeThreadId={activeThreadId}
        onSelectThread={setActiveThreadId}
        onNewChat={() => setActiveThreadId(null)}
        collapsed={sidebarCollapsed}
        onToggle={() => setSidebarCollapsed(!sidebarCollapsed)}
      />
      <main className="flex-1 overflow-hidden">
        <ChatWindow
          threadId={activeThreadId}
          onThreadCreated={setActiveThreadId}
        />
      </main>
    </div>
  );
}
