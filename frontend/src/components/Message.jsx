import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Bot, Check, Copy, RefreshCw, Square, ThumbsDown, ThumbsUp, User, Volume2 } from "lucide-react";
import Citations from "./Citations";
import { chatAPI } from "../api/chat";

/* ── Skeleton row ─────────────────────────────────────────── */
export function MessageSkeleton() {
  return (
    <div className="flex gap-4 bg-gray-50/80 px-4 py-4 dark:bg-gray-800/50">
      <div className="skeleton h-10 w-10 flex-shrink-0 rounded-xl" />
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
    <div className="animate-fade-in-up flex gap-4 bg-gray-50/80 px-4 py-4 dark:bg-gray-800/50">
      <div className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-xl bg-emerald-100 dark:bg-emerald-900/30 text-emerald-600 dark:text-emerald-400">
        <Bot className="h-6 w-6" />
      </div>
      <div className="flex flex-col">
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
      className={`animate-fade-in-up flex gap-4 px-4 py-4 ${
        isUser
          ? "bg-white dark:bg-gray-900"
          : "bg-gray-50/80 dark:bg-gray-800/50"
      }`}
    >
      {/* Avatar */}
      <div
        className={`flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-xl ${
          isUser
            ? "bg-primary-100 dark:bg-primary-900/30 text-primary-600 dark:text-primary-400"
            : "bg-emerald-100 dark:bg-emerald-900/30 text-emerald-600 dark:text-emerald-400"
        }`}
      >
        {isUser ? <User className="h-6 w-6" /> : <Bot className="h-6 w-6" />}
      </div>

      {/* Content */}
      <div className="min-w-0 flex-1">
        <p className="mb-2 text-sm font-medium text-gray-500 dark:text-gray-400">
          {isUser ? "Bạn" : "Minqes"}
        </p>

        {isUser ? (
          <p className="text-base md:text-lg leading-relaxed text-gray-800 dark:text-gray-200 whitespace-pre-wrap">
            {message.content}
          </p>
        ) : (
          <div className="prose prose-base md:prose-lg dark:prose-invert max-w-none text-gray-800 dark:text-gray-200">
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
          <div className="mt-3 flex items-center gap-2">
            {/* Speak */}
            <button
              onClick={handleSpeak}
              disabled={!speechSupported}
              className={`rounded-lg p-1.5 transition-colors ${
                isSpeaking
                  ? "bg-primary-100 text-primary-600 dark:bg-primary-900/30 dark:text-primary-300"
                  : "text-gray-400 dark:text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-700 hover:text-primary-600 dark:hover:text-primary-300"
              } disabled:cursor-not-allowed disabled:opacity-40`}
              title={isSpeaking ? "Dừng đọc" : "Đọc nội dung"}
            >
              {isSpeaking ? <Square className="h-6 w-6" /> : <Volume2 className="h-6 w-6" />}
            </button>

            {/* Copy */}
            <button
              onClick={handleCopy}
              className="rounded-lg p-1.5 text-gray-400 dark:text-gray-500 transition-colors hover:bg-gray-100 dark:hover:bg-gray-700 hover:text-gray-600 dark:hover:text-gray-300"
              title="Sao chép"
            >
              {copied
                ? <Check className="h-6 w-6 text-green-500" />
                : <Copy className="h-6 w-6" />}
            </button>

            {/* Regenerate */}
            {onRegenerate && (
              <button
                onClick={handleRegenerate}
                disabled={regenerating}
                className="rounded-lg p-1.5 text-gray-400 dark:text-gray-500 transition-colors hover:bg-gray-100 dark:hover:bg-gray-700 hover:text-gray-600 dark:hover:text-gray-300 disabled:opacity-40"
                title="Tạo lại"
              >
                <RefreshCw className={`h-6 w-6 ${regenerating ? "animate-spin" : ""}`} />
              </button>
            )}

            {/* Divider */}
            <span className="mx-0.5 h-6 w-px bg-gray-200 dark:bg-gray-700" />

            {/* Thumbs up */}
            <button
              onClick={() => handleFeedback("up")}
              disabled={!!feedbackGiven}
              className={`rounded p-1 transition-colors ${
                feedbackGiven === "up"
                  ? "text-green-500"
                  : "text-gray-400 dark:text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-700 hover:text-gray-600 dark:hover:text-gray-300"
              } disabled:cursor-default`}
              title="Hữu ích"
            >
              <ThumbsUp className="h-6 w-6" />
            </button>

            {/* Thumbs down */}
            <button
              onClick={() => handleFeedback("down")}
              disabled={!!feedbackGiven}
              className={`rounded p-1 transition-colors ${
                feedbackGiven === "down"
                  ? "text-red-500"
                  : "text-gray-400 dark:text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-700 hover:text-gray-600 dark:hover:text-gray-300"
              } disabled:cursor-default`}
              title="Chưa hữu ích"
            >
              <ThumbsDown className="h-6 w-6" />
            </button>
          </div>
        )}
      </div>
    </div>
  );
}


