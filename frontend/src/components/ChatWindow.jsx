import { useCallback, useEffect, useRef, useState } from "react";
import { chatAPI } from "../api/chat";
import Message, { MessageSkeleton, TypingIndicator } from "./Message";
import WebSearchToggle from "./WebSearchToggle";
import {
  ChevronDown,
  AlertCircle,
  BookOpen,
  Heart,
  Mic,
  Send,
  ShieldCheck,
  Stethoscope,
  X,
} from "lucide-react";

/* ── Error banner ─────────────────────────────────────────── */
function ErrorBanner({ message, onDismiss }) {
  if (!message) return null; 
  return (
    <div className="mx-4 mb-3 flex items-start gap-2 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-800 dark:bg-red-900/20 dark:text-red-400">
      <AlertCircle className="mt-0.5 h-6 w-6 flex-shrink-0" />
      <span className="flex-1">{message}</span>
      <button onClick={onDismiss} className="flex h-8 w-8 flex-shrink-0 items-center justify-center hover:opacity-70">
        <X className="h-6 w-6" />
      </button>
    </div>
  );
}

function ListeningIndicator({ visible, label }) {
  if (!visible) return null;

  return (
    <div className="mt-2 flex items-center gap-2 text-sm text-primary-600 dark:text-primary-400">
      <div className="flex items-end gap-0.5">
        {[0, 1, 2, 3].map((bar) => (
          <span
            key={bar}
            className="h-2.5 w-1 animate-pulse rounded-full bg-primary-500"
            style={{ animationDelay: `${bar * 120}ms` }}
          />
        ))}
      </div>
      <span>{label || "Đang nghe..."}</span>
    </div>
  );
}

/* ── Input area ───────────────────────────────────────────── */
function InputArea({
  value,
  onChange,
  onSend,
  disabled,
  isDictating,
  onToggleDictation,
  speechSupported,
  webSearchEnabled,
  onToggleWebSearch,
}) {
  const textareaRef = useRef(null);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height =
        Math.min(textareaRef.current.scrollHeight, 160) + "px";
    }
  }, [value]);

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      onSend();
    }
  };

  return (
    <div className="border-t border-gray-100 bg-white p-4 dark:border-gray-800 dark:bg-gray-900">
      <div className="mx-auto max-w-3xl">
        <div className="flex items-end gap-3 rounded-2xl border border-gray-200 bg-white px-4 py-3 shadow-sm transition-colors focus-within:border-primary-300 focus-within:shadow-md dark:border-gray-700 dark:bg-gray-800 dark:focus-within:border-primary-600">
          <textarea
            ref={textareaRef}
            value={value}
            onChange={(e) => onChange(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Nhập câu hỏi của bạn... (Enter để gửi, Shift+Enter xuống dòng)"
            rows={1}
            disabled={disabled}
            className="max-h-48 flex-1 resize-none bg-transparent py-2 text-base text-gray-800 placeholder-gray-400 outline-none disabled:opacity-60 dark:text-gray-200 dark:placeholder-gray-500"
          />
          <div className="mb-1 flex items-center gap-2">
            <WebSearchToggle
              enabled={webSearchEnabled}
              onToggle={onToggleWebSearch}
              disabled={disabled}
            />
            <button
              onClick={onToggleDictation}
              disabled={disabled || !speechSupported}
              title={speechSupported ? (isDictating ? "Dừng nhập giọng nói" : "Nhập bằng giọng nói") : "Trình duyệt chưa hỗ trợ nhập giọng nói"}
              className={`flex h-11 w-11 flex-shrink-0 items-center justify-center rounded-xl border transition-all ${
                isDictating
                  ? "animate-pulse border-red-400 bg-red-100 text-red-600 dark:border-red-500 dark:bg-red-900/30 dark:text-red-300"
                  : "border-gray-200 bg-white text-gray-600 hover:border-primary-300 hover:text-primary-600 dark:border-gray-700 dark:bg-gray-800 dark:text-gray-400 dark:hover:border-primary-600 dark:hover:text-primary-400"
              } disabled:cursor-not-allowed disabled:opacity-50`}
            >
              <Mic className="h-6 w-6" />
            </button>
            <button
              onClick={onSend}
              disabled={!value.trim() || disabled}
              className="flex h-11 w-11 flex-shrink-0 items-center justify-center rounded-xl bg-primary-600 text-white transition-all hover:bg-primary-700 disabled:cursor-not-allowed disabled:opacity-40"
              title="Gửi (Enter)"
            >
              <Send className="h-6 w-6" />
            </button>
          </div>
        </div>
        <ListeningIndicator visible={isDictating} label="Đang nghe..." />
        <p className="mt-2 text-center text-sm text-gray-400 dark:text-gray-600">
          Minqes cung cấp thông tin y khoa tham khảo. Luôn tham vấn bác sĩ cho các tình huống nghiêm trọng.
        </p>
      </div>
    </div>
  );
}

/* ── Main ChatWindow ──────────────────────────────────────── */
export default function ChatWindow({ threadId, onThreadCreated }) {
  const [messages, setMessages]           = useState([]);
  const [input, setInput]                 = useState("");
  const [sending, setSending]             = useState(false);
  const [loadingMessages, setLoadingMessages] = useState(false);
  const [error, setError]                 = useState(null);
  const [isDictating, setIsDictating]     = useState(false);
  const [speechSupported, setSpeechSupported] = useState(true);
  const [showScrollToBottom, setShowScrollToBottom] = useState(false);
  const [webSearchEnabled, setWebSearchEnabled] = useState(false);
  const messageListRef = useRef(null);
  const messagesEndRef = useRef(null);
  const speechRecognitionRef = useRef(null);
  const dictationBaseTextRef = useRef("");

  const handleSend = useCallback(async (contentOverride) => {
    const content = (contentOverride ?? input).trim();
    if (!content || sending) return;

    setError(null);
    let currentThreadId = threadId;

    // Auto-create thread if none exists
    if (!currentThreadId) {
      try {
        const title = content.length > 50 ? content.slice(0, 50) + "…" : content;
        const res = await chatAPI.createThread(title);
        currentThreadId = res.data.id;
        onThreadCreated(currentThreadId);
      } catch {
        setError("Không thể tạo cuộc trò chuyện. Vui lòng thử lại.");
        return;
      }
    }

    // Optimistic user message
    const tempId = "temp-" + Date.now();
    const tempUserMsg = {
      id: tempId,
      thread_id: currentThreadId,
      role: "user",
      content,
      created_at: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, tempUserMsg]);
    if (!contentOverride) setInput("");
    setSending(true);

    try {
      // ── Call /ask — one round-trip for user msg + RAG generation ──────────
      const res = await chatAPI.ask(currentThreadId, content, {
        web_search_enabled: webSearchEnabled,
      });
      const { user_message, assistant_message } = res.data;

      setMessages((prev) => [
        ...prev.filter((m) => m.id !== tempId),
        user_message,
        assistant_message,
      ]);
    } catch (err) {
      // Remove optimistic message on failure and restore input
      setMessages((prev) => prev.filter((m) => m.id !== tempId));
      if (!contentOverride) setInput(content);

      const status = err?.response?.status;
      if (status === 503) {
        setError("Dịch vụ AI đang không khả dụng. Vui lòng thử lại sau.");
      } else if (status === 408 || err?.code === "ECONNABORTED") {
        setError("Yêu cầu mất quá nhiều thời gian. Vui lòng thử câu hỏi ngắn hơn.");
      } else {
        setError("Đã có lỗi xảy ra khi xử lý yêu cầu. Vui lòng thử lại.");
      }
    } finally {
      setSending(false);
    }
  }, [input, sending, threadId, onThreadCreated, webSearchEnabled]);

  const handleToggleDictation = useCallback(() => {
    const recognition = speechRecognitionRef.current;
    if (!recognition) {
      setError("Trình duyệt chưa hỗ trợ nhập liệu bằng giọng nói.");
      return;
    }

    setError(null);
    if (isDictating) {
      recognition.stop();
      return;
    }

    dictationBaseTextRef.current = input?.trim() || "";
    recognition.start();
  }, [input, isDictating]);

  useEffect(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      setSpeechSupported(false);
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.lang = "vi-VN";
    recognition.interimResults = true;
    recognition.continuous = false;

    recognition.onstart = () => setIsDictating(true);
    recognition.onend = () => setIsDictating(false);
    recognition.onerror = (event) => {
      setIsDictating(false);
      if (event?.error === "not-allowed") {
        setError("Bạn chưa cấp quyền microphone. Vui lòng cho phép truy cập micro để dùng nhập giọng nói.");
      } else {
        setError("Không thể nhận diện giọng nói. Vui lòng thử lại.");
      }
    };

    recognition.onresult = (event) => {
      let finalTranscript = "";
      let interimTranscript = "";

      for (let i = event.resultIndex; i < event.results.length; i += 1) {
        const transcript = event.results[i]?.[0]?.transcript || "";
        if (event.results[i].isFinal) {
          finalTranscript += transcript;
        } else {
          interimTranscript += transcript;
        }
      }

      const spokenText = `${finalTranscript} ${interimTranscript}`.trim();
      const base = dictationBaseTextRef.current;
      setInput([base, spokenText].filter(Boolean).join(" ").trim());
    };

    speechRecognitionRef.current = recognition;

    return () => {
      recognition.onstart = null;
      recognition.onend = null;
      recognition.onerror = null;
      recognition.onresult = null;
      recognition.stop();
      speechRecognitionRef.current = null;
    };
  }, []);

  /* Load message history when thread changes */
  useEffect(() => {
    if (!threadId) { setMessages([]); return; }
    setLoadingMessages(true);
    setError(null);
    chatAPI
      .listMessages(threadId)
      .then((res) => setMessages(res.data))
      .catch(() => setError("Không thể tải lịch sử tin nhắn. Vui lòng thử lại."))
      .finally(() => setLoadingMessages(false));
  }, [threadId]);

  /* Auto-scroll */
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    setShowScrollToBottom(false);
  }, [messages, sending]);

  const updateScrollButtonVisibility = useCallback(() => {
    const el = messageListRef.current;
    if (!el) return;

    const distanceFromBottom = el.scrollHeight - (el.scrollTop + el.clientHeight);
    setShowScrollToBottom(distanceFromBottom > 300);
  }, []);

  const handleScrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  useEffect(() => {
    updateScrollButtonVisibility();
  }, [messages, updateScrollButtonVisibility]);

  /* Regenerate last assistant message */
  const handleRegenerate = useCallback(async (assistantMsg) => {
    // Find the user message that preceded this assistant message
    const msgIndex = messages.findIndex((m) => m.id === assistantMsg.id);
    if (msgIndex <= 0) return;
    const userMsg = messages[msgIndex - 1];
    if (!userMsg || userMsg.role !== "user") return;

    // Remove the old assistant reply and re-send the user question
    setMessages((prev) => prev.filter((m) => m.id !== assistantMsg.id));
    await handleSend(userMsg.content);
  }, [messages, handleSend]);

  /* ── Skeleton while loading history ─────────────────────── */
  if (loadingMessages) {
    return (
      <div className="flex h-full flex-col bg-white dark:bg-gray-900">
        <div className="flex-1 overflow-y-auto">
          {[...Array(4)].map((_, i) => (
            <MessageSkeleton key={i} />
          ))}
        </div>
        <InputArea
          value=""
          onChange={() => {}}
          onSend={() => {}}
          disabled
          isDictating={isDictating}
          onToggleDictation={handleToggleDictation}
          speechSupported={speechSupported}
          webSearchEnabled={webSearchEnabled}
          onToggleWebSearch={() => setWebSearchEnabled((v) => !v)}
        />
      </div>
    );
  }

  /* ── Welcome screen ──────────────────────────────────────── */
  if (!threadId && messages.length === 0) {
    return (
      <div className="flex h-full flex-col bg-white dark:bg-gray-900">
        <div className="flex flex-1 flex-col items-center justify-center px-4">
          <div className="mb-6 flex h-20 w-20 items-center justify-center rounded-3xl bg-primary-100 dark:bg-primary-900/30">
            <Heart className="h-12 w-12 text-primary-600 dark:text-primary-400" />
          </div>
          <h2 className="mb-2 text-2xl font-bold text-gray-900 dark:text-gray-100">
            Xin chào! 👋
          </h2>
          <p className="mb-8 max-w-md text-center text-gray-500 dark:text-gray-400">
            Tôi là Minqes — trợ lý y khoa AI. Hãy hỏi tôi về các vấn đề sức khỏe,
            triệu chứng, hoặc thông tin y khoa.
          </p>

          {/* Suggestion chips */}
          <div className="grid max-w-lg gap-3 sm:grid-cols-2">
            {[
              { icon: Stethoscope, text: "Triệu chứng đau đầu kéo dài có nguy hiểm không?" },
              { icon: BookOpen,    text: "Hướng dẫn sơ cứu khi bị bỏng" },
              { icon: ShieldCheck, text: "Cách phòng ngừa bệnh tim mạch" },
              { icon: Heart,       text: "Chế độ dinh dưỡng cho người tiểu đường" },
            ].map((item, i) => (
              <button
                key={i}
                onClick={() => setInput(item.text)}
                className="flex items-start gap-3 rounded-xl border border-gray-200 bg-white px-4 py-3 text-left text-sm text-gray-600 transition-all hover:border-primary-200 hover:bg-primary-50 hover:shadow-sm dark:border-gray-700 dark:bg-gray-800 dark:text-gray-300 dark:hover:border-primary-700 dark:hover:bg-primary-900/20"
              >
                <item.icon className="mt-0.5 h-6 w-6 flex-shrink-0 text-primary-500" />
                <span>{item.text}</span>
              </button>
            ))}
          </div>
        </div>

        <ErrorBanner message={error} onDismiss={() => setError(null)} />
        <InputArea
          value={input}
          onChange={setInput}
          onSend={handleSend}
          disabled={sending}
          isDictating={isDictating}
          onToggleDictation={handleToggleDictation}
          speechSupported={speechSupported}
          webSearchEnabled={webSearchEnabled}
          onToggleWebSearch={() => setWebSearchEnabled((v) => !v)}
        />
      </div>
    );
  }

  /* ── Active chat ─────────────────────────────────────────── */
  return (
    <div className="relative flex h-full flex-col bg-white dark:bg-gray-900">
      {/* Message list */}
      <div
        ref={messageListRef}
        onScroll={updateScrollButtonVisibility}
        className="flex-1 overflow-y-auto divide-y divide-gray-100 dark:divide-gray-800"
      >
        {messages.map((msg) => (
          <Message
            key={msg.id}
            message={msg}
            onRegenerate={msg.role === "assistant" ? handleRegenerate : undefined}
          />
        ))}

        {/* Typing indicator while waiting for response */}
        {sending && <TypingIndicator />}

        <div ref={messagesEndRef} />
      </div>

      {showScrollToBottom && (
        <button
          onClick={handleScrollToBottom}
          className="absolute bottom-28 left-1/2 z-20 flex h-11 w-11 -translate-x-1/2 items-center justify-center rounded-full border border-gray-200 bg-white text-primary-600 shadow-lg transition-all hover:-translate-y-0.5 hover:bg-primary-50 dark:border-gray-700 dark:bg-gray-800 dark:text-primary-300 dark:hover:bg-gray-700"
          title="Cuộn xuống tin nhắn mới nhất"
        >
          <ChevronDown className="h-6 w-6" />
        </button>
      )}

      <ErrorBanner message={error} onDismiss={() => setError(null)} />
      <InputArea
        value={input}
        onChange={setInput}
        onSend={handleSend}
        disabled={sending}
        isDictating={isDictating}
        onToggleDictation={handleToggleDictation}
        speechSupported={speechSupported}
        webSearchEnabled={webSearchEnabled}
        onToggleWebSearch={() => setWebSearchEnabled((v) => !v)}
      />
    </div>
  );
}

