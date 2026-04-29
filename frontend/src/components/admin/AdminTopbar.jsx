import { Moon, Sun, LogOut } from "lucide-react";
import { useTheme } from "../../contexts/ThemeContext";
import { useAuth } from "../../contexts/AuthContext";

export default function AdminTopbar() {
  const { isDark, toggle } = useTheme();
  const { user, logout } = useAuth();

  return (
    <header className="flex h-16 items-center justify-between border-b border-slate-800 bg-slate-950/80 px-6 backdrop-blur">
      <div>
        <h1 className="text-lg font-semibold text-slate-100">Admin Dashboard</h1>
        <p className="text-xs text-slate-400">Monitoring, analytics, and operations</p>
      </div>

      <div className="flex items-center gap-2">
        <button
          onClick={toggle}
          className="rounded-lg border border-slate-700 bg-slate-900 p-2 text-slate-300 hover:bg-slate-800"
          title="Toggle theme"
        >
          {isDark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
        </button>

        <div className="hidden rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-xs text-slate-300 md:block">
          {user?.name || user?.email}
        </div>

        <button
          onClick={logout}
          className="rounded-lg border border-slate-700 bg-slate-900 p-2 text-slate-300 hover:bg-slate-800"
          title="Sign out"
        >
          <LogOut className="h-4 w-4" />
        </button>
      </div>
    </header>
  );
}
