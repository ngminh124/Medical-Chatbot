import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { chatAPI } from "../api/chat";
import Message, { MessageSkeleton, TypingIndicator } from "./Message";
import WebSearchToggle from "./WebSearchToggle";
import { useSendMessage } from "../hooks/useSendMessage";
import {
  ChevronDown,
  AlertCircle,
  BookOpen,
  Heart,
  List,
  Mic,
  Square,
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
  onStopSending,
  disabled,
  isSending,
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
      if (isSending) return;
      onSend();
    }
  };

  const canSend = value.trim().length > 0 && !disabled && !isSending;

  return (
    <div className="px-3 pb-4 sm:px-6 sm:pb-6">
      <div className="mx-auto max-w-6xl">
        <div className="flex min-h-[60px] items-end gap-3 rounded-3xl border border-gray-300/60 bg-gray-100/95 px-4 py-3 shadow-[0_8px_28px_rgba(15,23,42,0.12)] backdrop-blur transition-colors dark:border-white/10 dark:bg-slate-800 dark:shadow-[0_12px_30px_rgba(0,0,0,0.45)]">
          <textarea
            ref={textareaRef}
            value={value}
            onChange={(e) => onChange(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Nhập câu hỏi của bạn... (Enter để gửi, Shift+Enter xuống dòng)"
            rows={1}
            disabled={disabled}
            className="max-h-48 flex-1 resize-none bg-transparent py-2.5 text-base text-gray-800 placeholder-gray-400 outline-none disabled:opacity-60 dark:text-gray-200 dark:placeholder-gray-500"
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
              onClick={isSending ? onStopSending : () => onSend()}
              disabled={!isSending && !canSend}
              className={`flex h-11 w-11 flex-shrink-0 items-center justify-center rounded-xl text-white transition-all ${
                isSending
                  ? "bg-red-500 hover:bg-red-600"
                  : "bg-primary-600 hover:bg-primary-700"
              } disabled:cursor-not-allowed disabled:bg-gray-300 disabled:text-gray-500 disabled:opacity-70 dark:disabled:bg-gray-700 dark:disabled:text-gray-400`}
              title={isSending ? "Dừng" : "Gửi (Enter)"}
            >
              {isSending ? <Square className="h-6 w-6" /> : <Send className="h-6 w-6" />}
            </button>
          </div>
        </div>
        <ListeningIndicator visible={isDictating} label="Đang nghe..." />
        <p className="mt-3 text-center text-sm text-gray-400 dark:text-gray-600">
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
  const [loadingMessages, setLoadingMessages] = useState(false);
  const [error, setError]                 = useState(null);
  const [isDictating, setIsDictating]     = useState(false);
  const [speechSupported, setSpeechSupported] = useState(true);
  const [showScrollToBottom, setShowScrollToBottom] = useState(false);
  const [activeQuestionId, setActiveQuestionId] = useState(null);
  const [desktopNavigatorOpen, setDesktopNavigatorOpen] = useState(false);
  const [mobileNavigatorOpen, setMobileNavigatorOpen] = useState(false);
  const [webSearchEnabled, setWebSearchEnabled] = useState(false);
  const messageListRef = useRef(null);
  const messagesEndRef = useRef(null);
  const messageRefs = useRef({});
  const speechRecognitionRef = useRef(null);
  const dictationBaseTextRef = useRef("");
  const desktopNavigatorCloseTimerRef = useRef(null);

  const performSend = useCallback(async (contentOverride, { signal } = {}) => {
    const content = (contentOverride ?? input).trim();
    if (!content) return;

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

    try {
      // ── Call /ask — one round-trip for user msg + RAG generation ──────────
      const res = await chatAPI.ask(currentThreadId, content, {
        web_search_enabled: webSearchEnabled,
      }, { signal });
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

      if (signal?.aborted || err?.name === "CanceledError" || err?.code === "ERR_CANCELED") {
        return;
      }

      const status = err?.response?.status;
      if (status === 503) {
        setError("Dịch vụ AI đang không khả dụng. Vui lòng thử lại sau.");
      } else if (status === 408 || err?.code === "ECONNABORTED") {
        setError("Yêu cầu mất quá nhiều thời gian. Vui lòng thử câu hỏi ngắn hơn.");
      } else {
        setError("Đã có lỗi xảy ra khi xử lý yêu cầu. Vui lòng thử lại.");
      }
    }
  }, [input, threadId, onThreadCreated, webSearchEnabled]);

  const {
    isSending,
    sendMessage,
    stopSending,
  } = useSendMessage(performSend);

  const handleSend = useCallback((contentOverride) => {
    sendMessage(contentOverride);
  }, [sendMessage]);

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
  }, [messages, isSending]);

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

  useEffect(() => {
    const handleResize = () => {
      updateScrollButtonVisibility();
    };

    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [updateScrollButtonVisibility]);

  const scrollToMessage = useCallback((messageId) => {
    const node = messageRefs.current[messageId];
    if (!node) return;
    setActiveQuestionId(messageId);
    node.scrollIntoView({ behavior: "smooth", block: "start" });
  }, []);

  const openDesktopNavigator = useCallback(() => {
    if (desktopNavigatorCloseTimerRef.current) {
      window.clearTimeout(desktopNavigatorCloseTimerRef.current);
      desktopNavigatorCloseTimerRef.current = null;
    }
    setDesktopNavigatorOpen(true);
  }, []);

  const closeDesktopNavigatorWithDelay = useCallback(() => {
    if (desktopNavigatorCloseTimerRef.current) {
      window.clearTimeout(desktopNavigatorCloseTimerRef.current);
    }
    desktopNavigatorCloseTimerRef.current = window.setTimeout(() => {
      setDesktopNavigatorOpen(false);
      desktopNavigatorCloseTimerRef.current = null;
    }, 220);
  }, []);

  useEffect(() => {
    return () => {
      if (desktopNavigatorCloseTimerRef.current) {
        window.clearTimeout(desktopNavigatorCloseTimerRef.current);
      }
    };
  }, []);

  const userQuestionItems = useMemo(
    () => messages
      .filter((msg) => msg.role === "user")
      .map((msg, index) => ({
        id: msg.id,
        label: `Câu hỏi ${index + 1}`,
        content: (msg.content || "").replace(/\s+/g, " ").trim(),
      })),
    [messages],
  );

  useEffect(() => {
    const root = messageListRef.current;
    if (!root || userQuestionItems.length === 0) {
      setActiveQuestionId(null);
      return;
    }

    const observedState = new Map();

    const pickActiveQuestionTopDown = () => {
      const intersecting = [];
      observedState.forEach((state, id) => {
        if (state.isIntersecting) {
          intersecting.push({ id, top: state.top, ratio: state.ratio });
        }
      });

      if (intersecting.length > 0) {
        const topSectionCandidates = intersecting
          .filter((item) => item.top >= 0)
          .sort((a, b) => a.top - b.top);

        const picked = topSectionCandidates[0]
          ?? intersecting.sort((a, b) => b.top - a.top)[0];

        if (picked?.id) {
          setActiveQuestionId((prev) => (prev === picked.id ? prev : picked.id));
        }
        return;
      }

      // Fallback: choose the latest question whose top has passed current scroll position.
      const currentScrollTop = root.scrollTop;
      const nearestPassed = userQuestionItems
        .map((item) => ({ id: item.id, node: messageRefs.current[item.id] }))
        .filter((item) => item.node)
        .filter((item) => item.node.offsetTop <= currentScrollTop + 12)
        .sort((a, b) => b.node.offsetTop - a.node.offsetTop)[0];

      const fallbackId = nearestPassed?.id ?? userQuestionItems[0]?.id ?? null;
      if (fallbackId) {
        setActiveQuestionId((prev) => (prev === fallbackId ? prev : fallbackId));
      }
    };

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          const id = entry.target.getAttribute("data-message-id");
          if (!id) return;
          const rootTop = entry.rootBounds?.top ?? 0;
          const relativeTop = entry.boundingClientRect.top - rootTop;
          observedState.set(id, {
            isIntersecting: entry.isIntersecting,
            ratio: entry.intersectionRatio,
            top: relativeTop,
          });
        });

        pickActiveQuestionTopDown();
      },
      {
        root,
        threshold: [0, 0.01, 0.1, 0.25, 0.5, 0.75, 1],
        rootMargin: "0px 0px -80% 0px",
      },
    );

    userQuestionItems.forEach((item) => {
      const node = messageRefs.current[item.id];
      if (node) {
        node.setAttribute("data-message-id", item.id);
        observer.observe(node);
      }
    });

    pickActiveQuestionTopDown();

    return () => observer.disconnect();
  }, [userQuestionItems]);

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
      <div className="flex h-full flex-col bg-white dark:bg-[#0f172a]">
        <div className="min-h-0 flex-1 overflow-y-auto">
          {[...Array(4)].map((_, i) => (
            <MessageSkeleton key={i} />
          ))}
        </div>
        <InputArea
          value=""
          onChange={() => {}}
          onSend={() => {}}
          onStopSending={() => {}}
          disabled
          isSending={false}
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
      <div className="flex h-full flex-col bg-white dark:bg-[#0f172a]">
        <div className="min-h-0 flex flex-1 flex-col items-center justify-center px-4">
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
          onStopSending={stopSending}
          disabled={isSending}
          isSending={isSending}
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
    <div className="relative h-full overflow-hidden bg-white dark:bg-[#0f172a]">
      <div className="relative mx-auto flex h-full w-full flex-col">
        {/* Message list */}
        <div
          ref={messageListRef}
          onScroll={updateScrollButtonVisibility}
          className="relative min-h-0 flex-1 overflow-y-auto bg-transparent px-2 pb-4 pt-6 sm:px-6 sm:pb-6 sm:pt-8"
        >
          <div className="mx-auto flex w-full max-w-6xl flex-col gap-8 sm:gap-10">
          {messages.map((msg) => (
            <div
              key={msg.id}
              ref={(el) => {
                if (el) {
                  messageRefs.current[msg.id] = el;
                } else {
                  delete messageRefs.current[msg.id];
                }
              }}
            >
              <Message
                message={msg}
                onRegenerate={msg.role === "assistant" ? handleRegenerate : undefined}
              />
            </div>
          ))}

          {/* Typing indicator while waiting for response */}
          {isSending && <TypingIndicator />}

          <div ref={messagesEndRef} />
          </div>
        </div>

        {showScrollToBottom && (
          <button
            onClick={handleScrollToBottom}
            className="absolute bottom-28 left-1/2 z-20 flex h-11 w-11 -translate-x-1/2 items-center justify-center rounded-full border border-gray-200 bg-white text-primary-600 shadow-lg transition-all hover:-translate-y-0.5 hover:bg-primary-50 dark:border-gray-700 dark:bg-gray-800 dark:text-primary-300 dark:hover:bg-gray-700 sm:bottom-32"
            title="Cuộn xuống tin nhắn mới nhất"
          >
            <ChevronDown className="h-6 w-6" />
          </button>
        )}

        <div className="sticky bottom-0 z-20 shrink-0 bg-white dark:bg-[#0f172a] relative">
          <div className="pointer-events-none absolute inset-x-0 -top-12 h-12 bg-gradient-to-t from-white to-transparent dark:from-[#0f172a]" />
          <div className="bg-white dark:bg-[#0f172a]">
            <ErrorBanner message={error} onDismiss={() => setError(null)} />
            <InputArea
              value={input}
              onChange={setInput}
              onSend={handleSend}
              onStopSending={stopSending}
              disabled={isSending}
              isSending={isSending}
              isDictating={isDictating}
              onToggleDictation={handleToggleDictation}
              speechSupported={speechSupported}
              webSearchEnabled={webSearchEnabled}
              onToggleWebSearch={() => setWebSearchEnabled((v) => !v)}
            />
          </div>
        </div>

        {/* Floating navigator (desktop) */}
        {userQuestionItems.length > 0 && (
          <>
            <div className="fixed right-4 top-1/2 z-30 hidden -translate-y-1/2 lg:block">
              <div
                className="relative flex items-center"
                onMouseEnter={openDesktopNavigator}
                onMouseLeave={closeDesktopNavigatorWithDelay}
              >
                <div className="max-h-[216px] w-[98px] overflow-y-auto rounded-2xl border border-white/10 bg-black/45 px-2 py-2 shadow-lg backdrop-blur-xl">
                  <div className="space-y-2">
                    {userQuestionItems.map((item) => {
                      const isActive = item.id === activeQuestionId;
                      return (
                        <button
                          key={item.id}
                          onClick={() => scrollToMessage(item.id)}
                          className={`group/nav relative flex w-full items-center justify-center rounded-md border-none bg-transparent py-1 transition-all ${
                            isActive ? "opacity-100" : "opacity-75 hover:opacity-100"
                          }`}
                          title={item.content}
                        >
                          <span
                            className={`h-2 w-12 rounded-full transition-all duration-200 ${
                              isActive
                                ? "bg-primary-400 shadow-[0_0_0_1px_rgba(255,255,255,0.25),0_0_16px_rgba(56,189,248,0.55)]"
                                : "bg-white/30 group-hover/nav:bg-white/60"
                            }`}
                          />
                          <span className="pointer-events-none absolute right-full mr-2 w-72 rounded-xl border border-white/10 bg-black/65 px-3 py-2 text-left text-xs text-white/90 opacity-0 shadow-xl backdrop-blur-xl transition-all duration-150 group-hover/nav:translate-x-0 group-hover/nav:opacity-100">
                            {item.content}
                          </span>
                        </button>
                      );
                    })}
                  </div>
                </div>

                <div className="absolute right-[92px] top-1/2 h-20 w-5 -translate-y-1/2 bg-transparent" />

                <div
                  className={`absolute right-[108px] top-1/2 max-h-[68vh] w-[320px] -translate-y-1/2 overflow-y-auto rounded-2xl border border-white/10 bg-black/60 p-3 shadow-xl backdrop-blur-xl transition-all duration-200 ${
                    desktopNavigatorOpen
                      ? "pointer-events-auto translate-x-0 opacity-100"
                      : "pointer-events-none translate-x-1 opacity-0"
                  }`}
                >
                  <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-white/70">
                    Mục lục câu hỏi
                  </p>
                  <div className="space-y-1.5">
                    {userQuestionItems.map((item) => (
                      <button
                        key={item.id}
                        onClick={() => scrollToMessage(item.id)}
                        className={`w-full rounded-xl border-l-2 px-3 py-2 text-left text-sm transition-all duration-200 ${
                          item.id === activeQuestionId
                            ? "border-l-primary-300 bg-white/16 text-white"
                            : "border-l-transparent bg-transparent text-white/80 hover:bg-white/10 hover:text-white"
                        }`}
                      >
                        <span className="block text-[11px] text-white/55">{item.label}</span>
                        <span className="line-clamp-2 block">{item.content}</span>
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* Mobile / tablet navigator toggle */}
            <div className="fixed bottom-28 right-4 z-30 lg:hidden">
              <button
                onClick={() => setMobileNavigatorOpen((v) => !v)}
                className="flex h-11 w-11 items-center justify-center rounded-full border border-white/10 bg-black/60 text-white shadow-lg backdrop-blur-md"
                title="Mở mục lục câu hỏi"
              >
                <List className="h-5 w-5" />
              </button>

              {mobileNavigatorOpen && (
                <div className="absolute bottom-14 right-0 max-h-[55vh] w-[280px] overflow-y-auto rounded-2xl border border-white/10 bg-black/70 p-3 shadow-xl backdrop-blur-xl">
                  <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-white/70">
                    Mục lục câu hỏi
                  </p>
                  <div className="space-y-1.5">
                    {userQuestionItems.map((item) => (
                      <button
                        key={item.id}
                        onClick={() => {
                          scrollToMessage(item.id);
                          setMobileNavigatorOpen(false);
                        }}
                        className={`w-full rounded-xl border-none px-3 py-2 text-left text-sm transition-colors ${
                          item.id === activeQuestionId
                            ? "bg-white/15 text-white"
                            : "bg-transparent text-white/85 hover:bg-white/10 hover:text-white"
                        }`}
                      >
                        <span className="block text-[11px] text-white/55">{item.label}</span>
                        <span className="line-clamp-2 block">{item.content}</span>
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

