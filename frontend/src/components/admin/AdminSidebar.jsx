import { NavLink } from "react-router-dom";
import {
  Activity,
  Bot,
  Cog,
  LayoutDashboard,
  MessageSquare,
  ScrollText,
  Users,
  Workflow,
} from "lucide-react";

const menu = [
  { to: "/admin", icon: LayoutDashboard, label: "Dashboard Overview", end: true },
  { to: "/admin/users", icon: Users, label: "Users Management" },
  { to: "/admin/conversations", icon: MessageSquare, label: "Conversations" },
  { to: "/admin/metrics", icon: Activity, label: "System Metrics" },
  { to: "/admin/logs", icon: ScrollText, label: "Logs (Loki)" },
  { to: "/admin/traces", icon: Workflow, label: "Traces (Tempo)" },
  { to: "/admin/settings", icon: Cog, label: "Settings" },
];

export default function AdminSidebar() {
  return (
    <aside className="hidden w-72 flex-col border-r border-slate-800 bg-slate-950 lg:flex">
      <div className="flex h-16 items-center gap-3 border-b border-slate-800 px-5">
        <div className="rounded-lg bg-primary-600 p-2">
          <Bot className="h-5 w-5 text-white" />
        </div>
        <div>
          <p className="text-sm font-semibold text-slate-100">Minqes Admin</p>
          <p className="text-xs text-slate-400">AI Medical Platform</p>
        </div>
      </div>

      <nav className="flex-1 space-y-1 p-3">
        {menu.map((item) => {
          const Icon = item.icon;
          return (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              className={({ isActive }) =>
                `flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm transition ${
                  isActive
                    ? "bg-primary-600/20 text-primary-300"
                    : "text-slate-300 hover:bg-slate-900 hover:text-slate-100"
                }`
              }
            >
              <Icon className="h-4 w-4" />
              <span>{item.label}</span>
            </NavLink>
          );
        })}
      </nav>
    </aside>
  );
}
