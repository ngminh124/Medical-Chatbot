import { Search, SearchX } from "lucide-react";

export default function WebSearchToggle({ enabled, onToggle, disabled = false }) {
  const tooltip = enabled
    ? "Đang bật tìm kiếm web – sẽ ưu tiên kết quả từ internet"
    : "Chỉ tìm trong cơ sở dữ liệu y tế (RAG)";

  return (
    <div className="group relative">
      <button
        type="button"
        onClick={onToggle}
        disabled={disabled}
        aria-pressed={enabled}
        title={tooltip}
        className={`flex h-11 items-center gap-2 rounded-xl border px-3 text-sm font-medium transition-all ${
          enabled
            ? "border-emerald-300 bg-emerald-50 text-emerald-700 hover:bg-emerald-100 dark:border-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300"
            : "border-gray-200 bg-white text-gray-600 hover:border-gray-300 hover:bg-gray-50 dark:border-gray-700 dark:bg-gray-800 dark:text-gray-300 dark:hover:bg-gray-700"
        } disabled:cursor-not-allowed disabled:opacity-50`}
      >
        {enabled ? <Search className="h-4 w-4" /> : <SearchX className="h-4 w-4" />}
        <span className="hidden sm:inline">Web Search</span>
      </button>

      <div className="pointer-events-none absolute bottom-full left-1/2 z-30 mb-2 hidden w-64 -translate-x-1/2 rounded-lg border border-gray-200 bg-white px-3 py-2 text-xs text-gray-600 shadow-lg group-hover:block dark:border-gray-700 dark:bg-gray-800 dark:text-gray-300">
        {tooltip}
      </div>
    </div>
  );
}
