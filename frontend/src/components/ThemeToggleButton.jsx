import { Moon, Sun } from "lucide-react";
import { useTheme } from "../contexts/ThemeContext";

export default function ThemeToggleButton({ className = "" }) {
  const { isDark, toggle } = useTheme();

  return (
    <button
      type="button"
      onClick={toggle}
      aria-label={isDark ? "Chuyen sang che do sang" : "Chuyen sang che do toi"}
      title={isDark ? "Che do sang" : "Che do toi"}
      className={`fixed right-5 top-5 z-40 flex h-11 w-11 items-center justify-center rounded-full border border-slate-200 bg-white/90 text-slate-600 shadow-lg backdrop-blur transition hover:-translate-y-0.5 hover:border-primary-300 hover:text-primary-600 dark:border-slate-700 dark:bg-slate-900/80 dark:text-slate-300 dark:hover:border-primary-500 dark:hover:text-primary-300 ${className}`}
    >
      {isDark ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
    </button>
  );
}
