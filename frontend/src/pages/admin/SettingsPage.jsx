import { useEffect, useState } from "react";
import { adminAPI } from "../../api/admin";

export default function SettingsPage() {
  const [settings, setSettings] = useState({
    rewrite_enabled: true,
    rerank_enabled: true,
    max_tokens: 512,
    top_k: 5,
  });
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    adminAPI.getSettings().then((res) => setSettings(res.data));
  }, []);

  const update = (key, value) => setSettings((prev) => ({ ...prev, [key]: value }));

  const save = async () => {
    setSaving(true);
    try {
      const res = await adminAPI.updateSettings(settings);
      setSettings(res.data);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="max-w-2xl rounded-2xl border border-slate-800 bg-slate-900/80 p-5">
      <h3 className="text-base font-semibold text-slate-100">RAG Runtime Settings</h3>

      <div className="mt-4 space-y-4">
        <label className="flex items-center justify-between rounded-xl border border-slate-800 bg-slate-950/60 p-3">
          <span className="text-sm text-slate-200">Enable rewrite</span>
          <input
            type="checkbox"
            checked={settings.rewrite_enabled}
            onChange={(e) => update("rewrite_enabled", e.target.checked)}
          />
        </label>

        <label className="flex items-center justify-between rounded-xl border border-slate-800 bg-slate-950/60 p-3">
          <span className="text-sm text-slate-200">Enable rerank</span>
          <input
            type="checkbox"
            checked={settings.rerank_enabled}
            onChange={(e) => update("rerank_enabled", e.target.checked)}
          />
        </label>

        <label className="block rounded-xl border border-slate-800 bg-slate-950/60 p-3">
          <span className="text-sm text-slate-200">Max tokens</span>
          <input
            type="number"
            min={128}
            max={4096}
            value={settings.max_tokens}
            onChange={(e) => update("max_tokens", Number(e.target.value))}
            className="mt-2 w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-slate-100"
          />
        </label>

        <label className="block rounded-xl border border-slate-800 bg-slate-950/60 p-3">
          <span className="text-sm text-slate-200">Top-k retrieval</span>
          <input
            type="number"
            min={1}
            max={20}
            value={settings.top_k}
            onChange={(e) => update("top_k", Number(e.target.value))}
            className="mt-2 w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-slate-100"
          />
        </label>
      </div>

      <button
        onClick={save}
        disabled={saving}
        className="mt-5 rounded-xl bg-primary-600 px-4 py-2 text-sm font-medium text-white hover:bg-primary-500 disabled:opacity-60"
      >
        {saving ? "Saving..." : "Save settings"}
      </button>
    </div>
  );
}
