import { useEffect, useState } from "react";
import { adminAPI } from "../../api/admin";
import MetricLineChart from "../../components/admin/MetricLineChart";

function toSeries(values, key) {
  return (values || []).map(({ timestamp, value }) => ({
    time: new Date(Number(timestamp) * 1000).toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    }),
    [key]: Number(value),
  }));
}

function mergeSeries(seriesList) {
  return seriesList
    .flat()
    .reduce((acc, item) => {
      const prev = acc.get(item.time) || { time: item.time };
      acc.set(item.time, { ...prev, ...item });
      return acc;
    }, new Map());
}

export default function MetricsPage() {
  const [data, setData] = useState([]);

  useEffect(() => {
    let active = true;

    const fetchData = async () => {
      try {
        const end = Math.floor(Date.now() / 1000);
        const start = end - 1800;

        const [rag, llm, retrieval] = await Promise.all([
          adminAPI.queryMetricsRange({
            query: "sum(rate(rag_request_duration_seconds_sum[5m])) / clamp_min(sum(rate(rag_request_duration_seconds_count[5m])), 1)",
            start,
            end,
            step: "5s",
          }),
          adminAPI.queryMetricsRange({
            query: "sum(rate(rag_llm_duration_seconds_sum[5m])) / clamp_min(sum(rate(rag_llm_duration_seconds_count[5m])), 1)",
            start,
            end,
            step: "5s",
          }),
          adminAPI.queryMetricsRange({
            query: "sum(rate(rag_retrieval_duration_seconds_sum[5m])) / clamp_min(sum(rate(rag_retrieval_duration_seconds_count[5m])), 1)",
            start,
            end,
            step: "5s",
          }),
        ]);

        const ragSeries = rag.data?.points || [];
        const llmSeries = llm.data?.points || [];
        const retrievalSeries = retrieval.data?.points || [];

        const merged = mergeSeries([
          toSeries(ragSeries, "rag_total_time_seconds"),
          toSeries(llmSeries, "llm_time_seconds"),
          toSeries(retrievalSeries, "retrieval_time_seconds"),
        ]);

        if (active) {
          setData(Array.from(merged.values()));
        }
      } catch {
        if (active) {
          setData([]);
        }
      }
    };

    fetchData();
    const timer = setInterval(fetchData, 5000);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, []);

  return (
    <MetricLineChart
      title="System Metrics (auto refresh 5s)"
      data={data}
      lines={[
        { key: "rag_total_time_seconds", label: "RAG total", color: "#38bdf8" },
        { key: "llm_time_seconds", label: "LLM", color: "#22c55e" },
        { key: "retrieval_time_seconds", label: "Retrieval", color: "#f59e0b" },
      ]}
      yFormatter={(v) => `${v.toFixed(2)}s`}
    />
  );
}
