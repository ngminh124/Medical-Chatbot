import { useEffect, useState } from "react";
import { adminAPI } from "../../api/admin";

export default function TracesPage() {
  const [service, setService] = useState("rag_pipeline");
  const [traces, setTraces] = useState([]);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;

    const load = () => {
      adminAPI
        .getTraces({ service, limit: 30 })
        .then((res) => {
          if (!active) return;
          setError("");
          setTraces(res.data.items || []);
        })
        .catch((err) => {
          if (!active) return;
          setTraces([]);
          setError(err.response?.data?.detail || "Cannot load traces");
        });
    };

    load();
    const timer = setInterval(load, 5000);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, [service]);

  return (
    <div className="space-y-4">
      <div className="rounded-2xl border border-slate-800 bg-slate-900/80 p-4">
        <label className="text-xs text-slate-400">Service name</label>
        <input
          value={service}
          onChange={(e) => setService(e.target.value)}
          placeholder="rag_pipeline"
          className="mt-1 w-full max-w-sm rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100"
        />
      </div>

      {error && (
        <div className="rounded-xl border border-rose-900 bg-rose-950/30 p-3 text-sm text-rose-200">
          {error}
        </div>
      )}

      <div className="rounded-2xl border border-slate-800 bg-slate-900/80 p-3">
        <p className="mb-3 text-sm text-slate-300">Trace timeline (rewrite → retrieval → rerank → llm)</p>
        <div className="space-y-2">
          {traces.map((trace, idx) => {
            const traceId = trace.traceID || trace.traceId || trace.trace_id || `trace-${idx}`;
            const start = trace.startTimeUnixNano || trace.startTime;
            const durationMs = Number(trace.durationMs || trace.duration || 0);
            return (
              <div key={traceId} className="rounded-xl border border-slate-800 bg-slate-950/60 p-3">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <p className="font-mono text-xs text-sky-300">{traceId}</p>
                  <span className="text-xs text-slate-500">{durationMs} ms</span>
                </div>
                <div className="mt-2 flex gap-2">
                  {["rewrite", "retrieval", "rerank", "llm"].map((step) => (
                    <div key={step} className="rounded-md border border-slate-700 bg-slate-900 px-2 py-1 text-[11px] text-slate-300">
                      {step}
                    </div>
                  ))}
                </div>
                {start && (
                  <p className="mt-2 text-xs text-slate-500">Start: {String(start)}</p>
                )}
              </div>
            );
          })}
          {!traces.length && <p className="text-sm text-slate-500">No traces available.</p>}
        </div>
      </div>
    </div>
  );
}
