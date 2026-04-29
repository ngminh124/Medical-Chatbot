import { useEffect, useMemo, useState } from "react";
import { adminAPI } from "../../api/admin";

function highlight(line) {
  return line
    .replace(/\[PERF\]/g, '<span class="text-cyan-300">[PERF]</span>')
    .replace(/\[ERROR\]/g, '<span class="text-rose-400">[ERROR]</span>')
    .replace(/\[CACHE\]/g, '<span class="text-amber-300">[CACHE]</span>');
}

export default function LogsPage() {
  const [level, setLevel] = useState("");
  const [keyword, setKeyword] = useState("");
  const [logs, setLogs] = useState([]);

  useEffect(() => {
    const load = () => {
      adminAPI
        .getLogs({ level, keyword, limit: 300 })
        .then((res) => {
          setLogs(res.data.items || []);
        })
        .catch(() => {
          setLogs([]);
        });
    };

    load();
    const interval = setInterval(load, 5000);
    return () => clearInterval(interval);
  }, [level, keyword]);

  const rows = useMemo(() => logs.slice(0, 300), [logs]);

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3 rounded-2xl border border-slate-800 bg-slate-900/80 p-3">
        <select
          value={level}
          onChange={(e) => setLevel(e.target.value)}
          className="rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-200"
        >
          <option value="">All levels</option>
          <option value="INFO">INFO</option>
          <option value="WARNING">WARNING</option>
          <option value="ERROR">ERROR</option>
        </select>

        <input
          value={keyword}
          onChange={(e) => setKeyword(e.target.value)}
          placeholder="Filter by keyword"
          className="min-w-[280px] flex-1 rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100 outline-none focus:border-primary-500"
        />
        <button
          onClick={() => setKeyword("ERROR")}
          className="rounded-md border border-slate-700 bg-slate-950 px-2 py-1 text-xs text-rose-300"
        >
          ERROR
        </button>
        <button
          onClick={() => setKeyword("REWRITE")}
          className="rounded-md border border-slate-700 bg-slate-950 px-2 py-1 text-xs text-amber-300"
        >
          REWRITE
        </button>
        <button
          onClick={() => setKeyword("LLM")}
          className="rounded-md border border-slate-700 bg-slate-950 px-2 py-1 text-xs text-sky-300"
        >
          LLM
        </button>
        <span className="text-xs text-slate-500">Live refresh: 5s</span>
      </div>

      <div className="rounded-2xl border border-slate-800 bg-slate-950/80 p-3">
        <div className="max-h-[70vh] space-y-2 overflow-auto font-mono text-xs">
          {rows.map((item, idx) => (
            <div key={`${item.ts}-${idx}`} className="rounded border border-slate-800 bg-slate-900/80 p-2 text-slate-300">
              <div className="mb-1 text-[11px] text-slate-500">
                {new Date(Number(item.ts) / 1e6).toLocaleString()}
              </div>
              <div
                dangerouslySetInnerHTML={{ __html: highlight(item.line) }}
                className="leading-5"
              />
            </div>
          ))}
          {!rows.length && <p className="text-slate-500">No logs found.</p>}
        </div>
      </div>
    </div>
  );
}
