import { ExternalLink, FileText } from "lucide-react";

function isValidUrl(url) {
  if (!url) return false;
  try {
    new URL(url);
    return true;
  } catch {
    return false;
  }
}

function extractDomain(url) {
  if (!isValidUrl(url)) return "";
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return "";
  }
}

function normalizeCitation(citation = {}) {
  const sourceAsUrl = isValidUrl(citation.source) ? citation.source : "";
  const url = citation.url || citation.link || sourceAsUrl || "";
  return {
    title: citation.title || citation.document_name || "Tài liệu tham khảo",
    snippet: citation.snippet || citation.content || citation.text || "",
    url,
    type: (citation.type || (url ? "web" : "rag")).toLowerCase(),
    score: typeof citation.score === "number" ? citation.score : null,
  };
}

function CitationBadge({ type }) {
  const isWeb = type === "web";
  return (
    <span
      className={`inline-flex items-center rounded-full px-2 py-0.5 text-[10px] font-semibold ${
        isWeb
          ? "bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-300"
          : "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300"
      }`}
    >
      {isWeb ? "Web" : "RAG"}
    </span>
  );
}

function CitationPreview({ citation }) {
  return (
    <div className="pointer-events-none absolute -top-2 right-0 z-30 hidden w-80 -translate-y-full rounded-xl border border-gray-200 bg-white p-3 text-xs text-gray-600 shadow-xl group-hover:block dark:border-gray-700 dark:bg-gray-900 dark:text-gray-300">
      <p className="font-semibold text-gray-800 dark:text-gray-100">{citation.title}</p>
      {citation.snippet && <p className="mt-1 line-clamp-5">{citation.snippet}</p>}
      {citation.url && (
        <p className="mt-2 truncate text-primary-600 dark:text-primary-400">{citation.url}</p>
      )}
    </div>
  );
}

export default function Citations({ citations = [], citationPrefix = "citation", onSelectCitation }) {
  if (!citations.length) return null;

  const normalized = citations.map(normalizeCitation);

  return (
    <div className="mt-4 rounded-xl border border-blue-100 bg-blue-50/60 p-3 dark:border-blue-900/50 dark:bg-blue-900/10">
      <p className="mb-3 flex items-center gap-2 text-xs font-semibold text-blue-700 dark:text-blue-300">
        <FileText className="h-3.5 w-3.5" />
        Nguồn tham khảo
      </p>

      <div className="space-y-2">
        {normalized.map((citation, i) => {
          const domain = extractDomain(citation.url);
          const index = i + 1;

          return (
            <article
              key={index}
              id={`${citationPrefix}-${index}`}
              className="group relative rounded-xl border border-gray-200 bg-white p-3 text-sm text-gray-700 shadow-sm transition-all hover:-translate-y-0.5 hover:shadow-md dark:border-gray-700 dark:bg-gray-800 dark:text-gray-200"
            >
              <CitationPreview citation={citation} />

              <div className="flex items-start gap-3">
                <button
                  type="button"
                  onClick={() => onSelectCitation?.(index)}
                  className="mt-0.5 inline-flex h-6 w-6 flex-shrink-0 items-center justify-center rounded-md bg-blue-100 text-xs font-bold text-blue-700 transition-colors hover:bg-blue-200 dark:bg-blue-900/40 dark:text-blue-300"
                >
                  {index}
                </button>

                <div className="min-w-0 flex-1">
                  <div className="mb-1 flex items-center gap-2">
                    <CitationBadge type={citation.type} />
                    {citation.score !== null && citation.score > 0 && (
                      <span className="text-[10px] text-gray-500 dark:text-gray-400">
                        score {(citation.score * 100).toFixed(0)}%
                      </span>
                    )}
                  </div>

                  {citation.url ? (
                    <a
                      href={citation.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-1 font-semibold text-gray-900 hover:text-primary-600 dark:text-gray-100 dark:hover:text-primary-400"
                    >
                      <span className="line-clamp-1">{citation.title}</span>
                      <ExternalLink className="h-3.5 w-3.5 flex-shrink-0" />
                    </a>
                  ) : (
                    <p className="font-semibold text-gray-900 dark:text-gray-100">{citation.title}</p>
                  )}

                  {domain && (
                    <p className="mt-0.5 text-xs text-gray-500 dark:text-gray-400">{domain}</p>
                  )}

                  {citation.snippet && (
                    <p className="mt-1.5 line-clamp-3 text-xs text-gray-600 dark:text-gray-300">
                      {citation.snippet}
                    </p>
                  )}
                </div>
              </div>
            </article>
          );
        })}
      </div>
    </div>
  );
}
