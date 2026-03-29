import { Children, cloneElement, isValidElement, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Bot, Check, Copy, RefreshCw, Square, ThumbsDown, ThumbsUp, User, Volume2 } from "lucide-react";
import Citations from "./Citations";
import { chatAPI } from "../api/chat";

/* ── Skeleton row ─────────────────────────────────────────── */
export function MessageSkeleton() {
  return (
    <div className="mx-auto flex w-full max-w-6xl gap-4 px-3 py-1 sm:px-4">
      <div className="skeleton h-10 w-10 flex-shrink-0 rounded-2xl" />
      <div className="flex-1 space-y-3 pt-1">
        <div className="skeleton h-4 w-32 rounded" />
        <div className="skeleton h-5 w-full rounded" />
        <div className="skeleton h-5 w-5/6 rounded" />
        <div className="skeleton h-5 w-3/4 rounded" />
      </div>
    </div>
  );
}

/* ── Typing indicator ─────────────────────────────────────── */
export function TypingIndicator() {
  return (
    <div className="animate-fade-in-up mx-auto flex w-full max-w-6xl gap-4 px-3 py-1 sm:px-4">
      <div className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-2xl bg-emerald-100 dark:bg-emerald-900/30 text-emerald-600 dark:text-emerald-400">
        <Bot className="h-6 w-6" />
      </div>
      <div className="flex flex-col px-1 py-1">
        <p className="mb-2 text-sm font-medium text-gray-500 dark:text-gray-400">
          Minqes
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
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [speechSupported, setSpeechSupported] = useState(true);
  const utteranceRef = useRef(null);
  const citationPrefix = `citation-${message.id}`;

  const rawCitations = message?.metadata_?.citations || [];
  const normalizedCitations = rawCitations.map((citation = {}) => {
    const sourceAsUrl =
      typeof citation.source === "string" && /^https?:\/\//.test(citation.source)
        ? citation.source
        : "";
    return {
      title: citation.title || citation.document_name || "Tài liệu tham khảo",
      url: citation.url || citation.link || sourceAsUrl || "",
      snippet: citation.snippet || citation.content || citation.text || "",
      type: (citation.type || (citation.url ? "web" : "rag")).toLowerCase(),
      score: typeof citation.score === "number" ? citation.score : null,
    };
  });

  const scrollToCitation = (index) => {
    const id = `${citationPrefix}-${index}`;
    const el = document.getElementById(id);
    if (!el) return;
    el.scrollIntoView({ behavior: "smooth", block: "nearest" });
    el.classList.add("ring-2", "ring-primary-300");
    window.setTimeout(() => {
      el.classList.remove("ring-2", "ring-primary-300");
    }, 900);
  };

  const CitationRef = ({ index }) => {
    const citation = normalizedCitations[index - 1];
    if (!citation) return `[${index}]`;

    return (
      <span className="group relative inline-flex align-super">
        <button
          type="button"
          onClick={() => scrollToCitation(index)}
          className="ml-0.5 rounded px-1 text-[10px] font-semibold text-sky-700 transition-colors hover:bg-sky-100 dark:text-sky-300 dark:hover:bg-sky-900/40"
          aria-label={`Xem nguồn tham khảo ${index}`}
        >
          [{index}]
        </button>
        <div className="pointer-events-none absolute bottom-full left-1/2 z-20 mb-2 hidden w-72 -translate-x-1/2 rounded-lg border border-gray-200 bg-white p-2 text-left text-xs text-gray-600 shadow-xl group-hover:block dark:border-gray-700 dark:bg-gray-900 dark:text-gray-300">
          <p className="font-semibold text-gray-800 dark:text-gray-100">[{index}] {citation.title}</p>
          {citation.snippet && <p className="mt-1 line-clamp-4">{citation.snippet}</p>}
          {citation.url && <p className="mt-1 truncate text-primary-600 dark:text-primary-400">{citation.url}</p>}
        </div>
      </span>
    );
  };

  const replaceCitationRefs = (node) => {
    if (typeof node === "string") {
      const parts = node.split(/(\[\d+\])/g);
      return parts.map((part, idx) => {
        const match = part.match(/^\[(\d+)\]$/);
        if (!match) return part;
        const index = Number(match[1]);
        if (!Number.isFinite(index) || index < 1 || index > normalizedCitations.length) {
          return part;
        }
        return <CitationRef key={`cite-ref-${idx}-${index}`} index={index} />;
      });
    }

    if (Array.isArray(node)) {
      return node.map((child) => replaceCitationRefs(child));
    }

    if (isValidElement(node)) {
      const replacedChildren = replaceCitationRefs(node.props.children);
      return cloneElement(node, { ...node.props, children: replacedChildren });
    }

    return node;
  };

  const MarkdownBlock = ({ children }) => {
    const replaced = replaceCitationRefs(children);
    return <>{Children.map(replaced, (child) => child)}</>;
  };

  useEffect(() => {
    const supported = typeof window !== "undefined" && "speechSynthesis" in window;
    setSpeechSupported(supported);
  }, []);

  const normalizeForSpeech = (text) =>
    (text || "")
      .replace(/```[\s\S]*?```/g, " ")
      .replace(/`[^`]*`/g, " ")
      .replace(/\[(.*?)\]\((.*?)\)/g, "$1")
      .replace(/[>#*_~]/g, " ")
      .replace(/\s+/g, " ")
      .trim();

  const handleSpeak = () => {
    if (!speechSupported) return;

    if (isSpeaking) {
      window.speechSynthesis.cancel();
      setIsSpeaking(false);
      return;
    }

    const textToSpeak = normalizeForSpeech(message.content);
    if (!textToSpeak) return;

    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(textToSpeak);
    utterance.lang = "vi-VN";
    utterance.rate = 1;
    utterance.pitch = 1;

    utterance.onstart = () => setIsSpeaking(true);
    utterance.onend = () => setIsSpeaking(false);
    utterance.onerror = () => setIsSpeaking(false);

    utteranceRef.current = utterance;
    window.speechSynthesis.speak(utterance);
  };

  useEffect(() => {
    return () => {
      if (utteranceRef.current && isSpeaking) {
        window.speechSynthesis.cancel();
      }
    };
  }, [isSpeaking]);

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
      className={`animate-fade-in-up mx-auto flex w-full max-w-6xl gap-4 px-3 py-1 sm:px-4 ${
        isUser ? "flex-row-reverse justify-start" : "justify-start"
      }`}
    >
      {/* Avatar */}
      <div
        className={`flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-2xl ${
          isUser
            ? "bg-primary-100 dark:bg-primary-900/30 text-primary-600 dark:text-primary-400"
            : "bg-emerald-100 dark:bg-emerald-900/30 text-emerald-600 dark:text-emerald-400"
        }`}
      >
        {isUser ? <User className="h-6 w-6" /> : <Bot className="h-6 w-6" />}
      </div>

      {/* Content */}
      <div
        className={`min-w-0 max-w-[88%] rounded-2xl px-4 py-3 sm:max-w-[82%] sm:px-5 sm:py-4 ${
          isUser
            ? "bg-primary-100 text-gray-900 dark:bg-primary-900/35 dark:text-gray-100"
            : "bg-transparent px-1 py-1"
        }`}
      >
        <p className="mb-2 text-sm font-medium text-gray-500 dark:text-gray-400">
          {isUser ? "Bạn" : "Minqes"}
        </p>

        {isUser ? (
          <p className="text-base md:text-lg leading-relaxed text-gray-800 dark:text-gray-200 whitespace-pre-wrap">
            {message.content}
          </p>
        ) : (
          <div className="prose prose-base md:prose-lg dark:prose-invert max-w-none text-gray-800 dark:text-gray-200">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                p: ({ children }) => <p><MarkdownBlock>{children}</MarkdownBlock></p>,
                li: ({ children }) => <li><MarkdownBlock>{children}</MarkdownBlock></li>,
                blockquote: ({ children }) => <blockquote><MarkdownBlock>{children}</MarkdownBlock></blockquote>,
                td: ({ children }) => <td><MarkdownBlock>{children}</MarkdownBlock></td>,
              }}
            >
              {message.content}
            </ReactMarkdown>
          </div>
        )}

        {/* Citations */}
        {!isUser && normalizedCitations.length > 0 && (
          <Citations
            citations={normalizedCitations}
            citationPrefix={citationPrefix}
            onSelectCitation={scrollToCitation}
          />
        )}

        {!isUser && message.metadata_ && normalizedCitations.length === 0 && (
          <p className="mt-3 text-xs text-gray-500 dark:text-gray-400">
            Không có tài liệu tham khảo cho câu trả lời này.
          </p>
        )}

        {/* Action buttons (assistant only) */}
        {!isUser && (
          <div className="mt-3 flex items-center gap-1.5">
            {/* Speak */}
            <button
              onClick={handleSpeak}
              disabled={!speechSupported}
              className={`flex h-9 w-9 items-center justify-center rounded-lg transition-colors ${
                isSpeaking
                  ? "bg-primary-100 text-primary-600 dark:bg-primary-900/30 dark:text-primary-300"
                  : "text-gray-400 dark:text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-700 hover:text-primary-600 dark:hover:text-primary-300"
              } disabled:cursor-not-allowed disabled:opacity-40`}
              title={isSpeaking ? "Dừng đọc" : "Đọc nội dung"}
            >
              {isSpeaking ? <Square className="h-5 w-5" /> : <Volume2 className="h-5 w-5" />}
            </button>

            {/* Copy */}
            <button
              onClick={handleCopy}
              className="flex h-9 w-9 items-center justify-center rounded-lg text-gray-400 dark:text-gray-500 transition-colors hover:bg-gray-100 dark:hover:bg-gray-700 hover:text-gray-600 dark:hover:text-gray-300"
              title="Sao chép"
            >
              {copied
                ? <Check className="h-5 w-5 text-green-500" />
                : <Copy className="h-5 w-5" />}
            </button>

            {/* Regenerate */}
            {onRegenerate && (
              <button
                onClick={handleRegenerate}
                disabled={regenerating}
                className="flex h-9 w-9 items-center justify-center rounded-lg text-gray-400 dark:text-gray-500 transition-colors hover:bg-gray-100 dark:hover:bg-gray-700 hover:text-gray-600 dark:hover:text-gray-300 disabled:opacity-40"
                title="Tạo lại"
              >
                <RefreshCw className={`h-5 w-5 ${regenerating ? "animate-spin" : ""}`} />
              </button>
            )}

            {/* Divider */}
            <span className="mx-1 h-5 w-px bg-gray-200 dark:bg-gray-700" />

            {/* Thumbs up */}
            <button
              onClick={() => handleFeedback("up")}
              disabled={!!feedbackGiven}
              className={`flex h-9 w-9 items-center justify-center rounded-lg transition-colors ${
                feedbackGiven === "up"
                  ? "text-green-500"
                  : "text-gray-400 dark:text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-700 hover:text-gray-600 dark:hover:text-gray-300"
              } disabled:cursor-default`}
              title="Hữu ích"
            >
              <ThumbsUp className="h-5 w-5" />
            </button>

            {/* Thumbs down */}
            <button
              onClick={() => handleFeedback("down")}
              disabled={!!feedbackGiven}
              className={`flex h-9 w-9 items-center justify-center rounded-lg transition-colors ${
                feedbackGiven === "down"
                  ? "text-red-500"
                  : "text-gray-400 dark:text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-700 hover:text-gray-600 dark:hover:text-gray-300"
              } disabled:cursor-default`}
              title="Chưa hữu ích"
            >
              <ThumbsDown className="h-5 w-5" />
            </button>
          </div>
        )}
      </div>
    </div>
  );
}


