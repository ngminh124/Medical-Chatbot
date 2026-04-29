import { useEffect, useMemo, useState } from "react";
import { adminAPI } from "../../api/admin";
import StatCard from "../../components/admin/StatCard";
import MetricLineChart from "../../components/admin/MetricLineChart";

function toChartMap(points, key) {
  const map = new Map();
  (points || []).forEach((point) => {
    const ts = point?.timestamp;
    const value = point?.value;
    if (value === undefined || Number.isNaN(Number(value))) return;
    const t = new Date(Number(ts) * 1000).toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
    map.set(t, { ...(map.get(t) || { time: t }), [key]: Number(value) });
  });
  return map;
}

export default function OverviewPage() {
  const [overview, setOverview] = useState(null);
  const [chartData, setChartData] = useState([]);
  const grafanaSrc = `${window.location.protocol}//${window.location.hostname}:3000/d/rag-observability/rag-observability-dashboard?orgId=1&from=now-1h&to=now&kiosk`;

  useEffect(() => {
    let active = true;

    const load = async () => {
      try {
        const now = Math.floor(Date.now() / 1000);
        const start = now - 3600;

        const [
          overviewRes,
          reqRes,
          llmP50Res,
          llmP95Res,
          retrievalRes,
          errorRes,
          tokensRes,
          activeReqRes,
        ] =
          await Promise.all([
            adminAPI.getOverview(),
            adminAPI.queryMetricsRange({
              query: "sum(increase(rag_requests_total[5m])) / 5",
              start,
              end: now,
              step: "30s",
            }),
            adminAPI.queryMetricsRange({
              query:
                "histogram_quantile(0.50, sum by (le) (rate(rag_llm_duration_seconds_bucket[5m])))",
              start,
              end: now,
              step: "30s",
            }),
            adminAPI.queryMetricsRange({
              query:
                "histogram_quantile(0.95, sum by (le) (rate(rag_llm_duration_seconds_bucket[5m])))",
              start,
              end: now,
              step: "30s",
            }),
            adminAPI.queryMetricsRange({
              query:
                "histogram_quantile(0.95, sum by (le) (rate(rag_retrieval_duration_seconds_bucket[5m])))",
              start,
              end: now,
              step: "30s",
            }),
            adminAPI.queryMetricsRange({
              query:
                "sum(rate(rag_errors_total[5m])) / clamp_min(sum(rate(rag_requests_total[5m])), 1)",
              start,
              end: now,
              step: "30s",
            }),
            adminAPI.queryMetricsRange({
              query: "sum(increase(rag_tokens_generated_total[5m])) / 300",
              start,
              end: now,
              step: "30s",
            }),
            adminAPI.queryMetricsRange({
              query: "sum(rag_active_requests)",
              start,
              end: now,
              step: "30s",
            }),
          ]);

        if (!active) return;
        setOverview(overviewRes.data || null);

        const reqPoints = reqRes.data?.points || [];
        const p50Points = llmP50Res.data?.points || [];
        const p95Points = llmP95Res.data?.points || [];
        const retrievalPoints = retrievalRes.data?.points || [];
        const errorPoints = errorRes.data?.points || [];
        const tokenPoints = tokensRes.data?.points || [];
        const activePoints = activeReqRes.data?.points || [];

        const combined = [
          ...toChartMap(reqPoints, "requests").values(),
          ...toChartMap(p50Points, "llmP50").values(),
          ...toChartMap(p95Points, "llmP95").values(),
          ...toChartMap(retrievalPoints, "retrievalP95").values(),
          ...toChartMap(errorPoints, "errorRate").values(),
          ...toChartMap(tokenPoints, "tokensPerSecond").values(),
          ...toChartMap(activePoints, "activeRequests").values(),
        ].reduce((acc, row) => {
          const old = acc.get(row.time) || { time: row.time };
          acc.set(row.time, { ...old, ...row });
          return acc;
        }, new Map());

        setChartData(Array.from(combined.values()));
      } catch {
        if (active) {
          setChartData([]);
        }
      }
    };

    load();
    const interval = setInterval(load, 5000);
    return () => {
      active = false;
      clearInterval(interval);
    };
  }, []);

  const cards = useMemo(
    () => [
      { label: "Total users", value: overview?.total_users ?? 0 },
      { label: "Active users today", value: overview?.active_users_today ?? 0 },
      { label: "Total conversations", value: overview?.total_conversations ?? 0 },
      {
        label: "Requests / minute",
        value:
          overview?.requests_per_minute !== undefined
            ? Number(overview.requests_per_minute).toFixed(2)
            : "0.00",
      },
      {
        label: "Tokens / second",
        value:
          overview?.tokens_per_second !== undefined
            ? Number(overview.tokens_per_second).toFixed(2)
            : "0.00",
      },
      {
        label: "Active requests",
        value:
          overview?.active_requests !== undefined
            ? Number(overview.active_requests).toFixed(0)
            : "0",
      },
      {
        label: "Cache hit rate",
        value:
          overview?.cache_hit_rate !== undefined
            ? (Number(overview.cache_hit_rate) * 100).toFixed(1)
            : "0.0",
        suffix: "%",
      },
      {
        label: "Avg response time",
        value:
          overview?.average_response_time !== undefined
            ? Number(overview.average_response_time).toFixed(3)
            : "0.000",
        suffix: "s",
      },
      {
        label: "Monitoring status",
        value:
          overview?.monitoring?.fastapi && overview?.monitoring?.prometheus
            ? "UP"
            : "PARTIAL",
      },
    ],
    [overview]
  );

  return (
    <div className="space-y-6">
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
        {cards.map((card) => (
          <StatCard key={card.label} {...card} />
        ))}
      </div>

      <div className="grid gap-6 xl:grid-cols-2">
        <MetricLineChart
          title="Requests over time"
          data={chartData}
          lines={[{ key: "requests", label: "RPS", color: "#38bdf8" }]}
        />
        <MetricLineChart
          title="LLM latency (P50 / P95)"
          data={chartData}
          lines={[
            { key: "llmP50", label: "P50", color: "#34d399" },
            { key: "llmP95", label: "P95", color: "#f59e0b" },
          ]}
          yFormatter={(v) => `${v}s`}
        />
        <MetricLineChart
          title="Retrieval latency (P95)"
          data={chartData}
          lines={[{ key: "retrievalP95", label: "Retrieval P95", color: "#a78bfa" }]}
          yFormatter={(v) => `${v}s`}
        />
        <MetricLineChart
          title="Error rate"
          data={chartData}
          lines={[{ key: "errorRate", label: "Error rate", color: "#f43f5e" }]}
          yFormatter={(v) => `${(v * 100).toFixed(1)}%`}
        />
        <MetricLineChart
          title="Token throughput"
          data={chartData}
          lines={[{ key: "tokensPerSecond", label: "Tokens/s", color: "#60a5fa" }]}
        />
        <MetricLineChart
          title="Active requests"
          data={chartData}
          lines={[{ key: "activeRequests", label: "In-flight", color: "#f97316" }]}
        />
      </div>

      <div className="rounded-2xl border border-slate-800 bg-slate-900/80 p-4 shadow-lg shadow-black/20">
        <h3 className="mb-3 text-sm font-medium text-slate-200">Grafana (embedded)</h3>
        <iframe
          title="Grafana RAG Dashboard"
          src={grafanaSrc}
          className="h-[520px] w-full rounded-lg border border-slate-800 bg-slate-950"
        />
      </div>
    </div>
  );
}
