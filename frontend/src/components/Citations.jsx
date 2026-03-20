import { ExternalLink, FileText } from "lucide-react";

export default function Citations({ metadata }) {
  if (!metadata?.citations?.length) return null;

  return (
    <div className="mt-3 rounded-lg border border-blue-100 dark:border-blue-900/50 bg-blue-50/50 dark:bg-blue-900/10 p-3">
      <p className="mb-2 flex items-center gap-1.5 text-xs font-semibold text-blue-700 dark:text-blue-400">
        <FileText className="h-3.5 w-3.5" />
        Nguồn tham khảo
      </p>
      <div className="space-y-1.5">
        {metadata.citations.map((citation, i) => (
          <div
            key={i}
            className="flex items-start gap-2 rounded-md bg-white dark:bg-gray-800 px-2.5 py-2 text-xs text-gray-600 dark:text-gray-300 shadow-sm"
          >
            <span className="mt-0.5 flex h-4 w-4 flex-shrink-0 items-center justify-center rounded bg-blue-100 dark:bg-blue-900/40 text-[10px] font-bold text-blue-600 dark:text-blue-400">
              {i + 1}
            </span>
            <div className="min-w-0 flex-1">
              <p className="font-medium text-gray-800 dark:text-gray-200">
                {citation.title || citation.document_name || "Tài liệu y khoa"}
              </p>
              {citation.content && (
                <p className="mt-0.5 line-clamp-2 text-gray-500 dark:text-gray-400">
                  {citation.content}
                </p>
              )}
              {citation.source && (
                <p className="mt-0.5 text-gray-400 dark:text-gray-500 truncate">
                  {citation.source}
                </p>
              )}
              {citation.score > 0 && (
                <p className="mt-0.5 text-gray-400 dark:text-gray-600 text-[10px]">
                  Điểm phù hợp: {(citation.score * 100).toFixed(0)}%
                </p>
              )}
            </div>
            {citation.url && (
              <a
                href={citation.url}
                target="_blank"
                rel="noopener noreferrer"
                className="flex-shrink-0 text-blue-500 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300"
              >
                <ExternalLink className="h-3.5 w-3.5" />
              </a>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
