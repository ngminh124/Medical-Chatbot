import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Bot, Check, Copy, RefreshCw, ThumbsDown, ThumbsUp, User } from "lucide-react";
import Citations from "./Citations";
import { chatAPI } from "../api/chat";

/* ── Skeleton row ─────────────────────────────────────────── */
export function MessageSkeleton() {
  return (
    <div className="flex gap-3 px-4 py-4 bg-gray-50/80 dark:bg-gray-800/50">
      <div className="skeleton h-8 w-8 flex-shrink-0 rounded-lg" />
      <div className="flex-1 space-y-2 pt-1">
        <div className="skeleton h-3 w-24 rounded" />
        <div className="skeleton h-4 w-full rounded" />
        <div className="skeleton h-4 w-5/6 rounded" />
        <div className="skeleton h-4 w-3/4 rounded" />
      </div>
    </div>
  );
}

/* ── Typing indicator ─────────────────────────────────────── */
export function TypingIndicator() {
  return (
    <div className="animate-fade-in-up flex gap-3 px-4 py-4 bg-gray-50/80 dark:bg-gray-800/50">
      <div className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-lg bg-emerald-100 dark:bg-emerald-900/30 text-emerald-600 dark:text-emerald-400">
        <Bot className="h-4 w-4" />
      </div>
      <div className="flex flex-col">
        <p className="mb-2 text-xs font-medium text-gray-500 dark:text-gray-400">
          Medical Assistant
        </p>
        <div className="flex items-center gap-1.5 text-gray-400 dark:text-gray-500">
          <span className="typing-dot" />
          <span className="typing-dot" />
          <span className="typing-dot" />
        </div>
      </div>
    </div>
  );
}

/* ── Main Message component ───────────────────────────────── */
export default function Message({ message, onRegenerate }) {
  const isUser = message.role === "user";
  const [copied, setCopied] = useState(false);
  const [feedbackGiven, setFeedbackGiven] = useState(null); // 'up' | 'down'
  const [regenerating, setRegenerating] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      /* clipboard blocked */
    }
  };

  const handleFeedback = async (type) => {
    if (feedbackGiven) return;
    try {
      await chatAPI.createFeedback(
        message.id,
        type === "up" ? 5 : 1,
        type === "up" ? "Hữu ích" : "Chưa hữu ích",
      );
      setFeedbackGiven(type);
    } catch {
      /* silent */
    }
  };

  const handleRegenerate = async () => {
    if (regenerating || !onRegenerate) return;
    setRegenerating(true);
    try {
      await onRegenerate(message);
    } finally {
      setRegenerating(false);
    }
  };

  return (
    <div
      className={`animate-fade-in-up flex gap-3 px-4 py-4 ${
        isUser
          ? "bg-white dark:bg-gray-900"
          : "bg-gray-50/80 dark:bg-gray-800/50"
      }`}
    >
      {/* Avatar */}
      <div
        className={`flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-lg ${
          isUser
            ? "bg-primary-100 dark:bg-primary-900/30 text-primary-600 dark:text-primary-400"
            : "bg-emerald-100 dark:bg-emerald-900/30 text-emerald-600 dark:text-emerald-400"
        }`}
      >
        {isUser ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
      </div>

      {/* Content */}
      <div className="min-w-0 flex-1">
        <p className="mb-1 text-xs font-medium text-gray-500 dark:text-gray-400">
          {isUser ? "Bạn" : "Medical Assistant"}
        </p>

        {isUser ? (
          <p className="text-sm leading-relaxed text-gray-800 dark:text-gray-200 whitespace-pre-wrap">
            {message.content}
          </p>
        ) : (
          <div className="prose prose-sm dark:prose-invert max-w-none text-gray-800 dark:text-gray-200">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {message.content}
            </ReactMarkdown>
          </div>
        )}

        {/* Citations */}
        {!isUser && message.metadata_?.citations?.length > 0 && (
          <Citations metadata={message.metadata_} />
        )}

        {/* Action buttons (assistant only) */}
        {!isUser && (
          <div className="mt-2 flex items-center gap-1">
            {/* Copy */}
            <button
              onClick={handleCopy}
              className="rounded p-1 text-gray-300 dark:text-gray-600 transition-colors hover:bg-gray-100 dark:hover:bg-gray-700 hover:text-gray-500 dark:hover:text-gray-300"
              title="Sao chép"
            >
              {copied
                ? <Check className="h-3.5 w-3.5 text-green-500" />
                : <Copy className="h-3.5 w-3.5" />}
            </button>

            {/* Regenerate */}
            {onRegenerate && (
              <button
                onClick={handleRegenerate}
                disabled={regenerating}
                className="rounded p-1 text-gray-300 dark:text-gray-600 transition-colors hover:bg-gray-100 dark:hover:bg-gray-700 hover:text-gray-500 dark:hover:text-gray-300 disabled:opacity-40"
                title="Tạo lại"
              >
                <RefreshCw className={`h-3.5 w-3.5 ${regenerating ? "animate-spin" : ""}`} />
              </button>
            )}

            {/* Divider */}
            <span className="mx-0.5 h-3.5 w-px bg-gray-200 dark:bg-gray-700" />

            {/* Thumbs up */}
            <button
              onClick={() => handleFeedback("up")}
              disabled={!!feedbackGiven}
              className={`rounded p-1 transition-colors ${
                feedbackGiven === "up"
                  ? "text-green-500"
                  : "text-gray-300 dark:text-gray-600 hover:bg-gray-100 dark:hover:bg-gray-700 hover:text-gray-500 dark:hover:text-gray-300"
              } disabled:cursor-default`}
              title="Hữu ích"
            >
              <ThumbsUp className="h-3.5 w-3.5" />
            </button>

            {/* Thumbs down */}
            <button
              onClick={() => handleFeedback("down")}
              disabled={!!feedbackGiven}
              className={`rounded p-1 transition-colors ${
                feedbackGiven === "down"
                  ? "text-red-500"
                  : "text-gray-300 dark:text-gray-600 hover:bg-gray-100 dark:hover:bg-gray-700 hover:text-gray-500 dark:hover:text-gray-300"
              } disabled:cursor-default`}
              title="Chưa hữu ích"
            >
              <ThumbsDown className="h-3.5 w-3.5" />
            </button>
          </div>
        )}
      </div>
    </div>
  );
}


