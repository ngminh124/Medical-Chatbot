import { useCallback, useEffect, useRef, useState } from "react";
import { chatAPI } from "../api/chat";
import Message, { MessageSkeleton, TypingIndicator } from "./Message";
import { useSpeechRecognition } from "../hooks/useSpeechRecognition";
import { useTextToSpeech } from "../hooks/useTextToSpeech";
import {
  AlertCircle,
  BookOpen,
  Heart,
  Loader2,
  MicOff,
  Send,
  ShieldCheck,
  Square,
  Stethoscope,
  Volume2,
  VolumeX,
  X,
} from "lucide-react";

/* ── Error banner ─────────────────────────────────────────── */
function ErrorBanner({ message, onDismiss }) {
  if (!message) return null;
  return (
    <div className="mx-4 mb-2 flex items-start gap-2 rounded-lg border border-red-200 dark:border-red-800 bg-red-50 dark:bg-red-900/20 px-4 py-3 text-sm text-red-700 dark:text-red-400">
      <AlertCircle className="mt-0.5 h-4 w-4 flex-shrink-0" />
      <span className="flex-1">{message}</span>
      <button onClick={onDismiss} className="flex-shrink-0 hover:opacity-70">
        <X className="h-4 w-4" />
      </button>
    </div>
  );
}

function IconToggleButton({
  active,
  disabled,
  onClick,
  title,
  icon: Icon,
  activeIcon: ActiveIcon,
  loading,
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      title={title}
      className={`mb-1 flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-lg border transition-all ${
        active
          ? "border-primary-400 bg-primary-100 text-primary-700 dark:border-primary-500 dark:bg-primary-900/40 dark:text-primary-300"
          : "border-gray-200 bg-white text-gray-500 hover:border-primary-300 hover:text-primary-600 dark:border-gray-700 dark:bg-gray-800 dark:text-gray-400 dark:hover:border-primary-600 dark:hover:text-primary-400"
      } disabled:cursor-not-allowed disabled:opacity-50`}
    >
      {loading ? (
        <Loader2 className="h-4 w-4 animate-spin" />
      ) : active && ActiveIcon ? (
        <ActiveIcon className="h-4 w-4" />
      ) : (
        <Icon className="h-4 w-4" />
      )}
    </button>
  );
}

function ListeningIndicator({ visible, label }) {
  if (!visible) return null;

  return (
    <div className="mt-2 flex items-center gap-2 text-xs text-primary-600 dark:text-primary-400">
      <div className="flex items-end gap-0.5">
        {[0, 1, 2, 3].map((bar) => (
          <span
            key={bar}
            className="h-1.5 w-1 animate-pulse rounded-full bg-primary-500"
            style={{ animationDelay: `${bar * 120}ms` }}
          />
        ))}
      </div>
      <span>{label || "Đang nghe..."}</span>
    </div>
  );
}

/* ── Input area ───────────────────────────────────────────── */
function InputArea({ value, onChange, onSend, disabled, speech, tts, onToggleSpeech, onToggleTts }) {
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
    <div className="border-t border-gray-100 dark:border-gray-800 bg-white dark:bg-gray-900 p-4">
      <div className="mx-auto max-w-3xl">
        <div className="flex items-end gap-2 rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 px-4 py-2 shadow-sm transition-colors focus-within:border-primary-300 dark:focus-within:border-primary-600 focus-within:shadow-md">
          <textarea
            ref={textareaRef}
            value={value}
            onChange={(e) => onChange(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Nhập câu hỏi của bạn... (Enter để gửi, Shift+Enter xuống dòng)"
            rows={1}
            disabled={disabled}
            className="max-h-40 flex-1 resize-none bg-transparent py-1.5 text-sm text-gray-800 dark:text-gray-200 placeholder-gray-400 dark:placeholder-gray-500 outline-none disabled:opacity-60"
          />
          <IconToggleButton
            active={speech.isRecording}
            loading={speech.isTranscribing}
            icon={MicOff}
            activeIcon={Square}
            title={speech.isRecording ? "Dừng ghi âm" : "Nhập bằng giọng nói"}
            onClick={onToggleSpeech}
            disabled={disabled || speech.isTranscribing}
          />
          <IconToggleButton
            active={tts.enabled || tts.isPlaying}
            loading={tts.isLoading}
            icon={VolumeX}
            activeIcon={tts.isPlaying ? Square : Volume2}
            title={tts.isPlaying ? "Dừng phát âm" : "Bật/tắt đọc phản hồi"}
            onClick={onToggleTts}
            disabled={disabled || speech.isTranscribing}
          />
          <button
            onClick={onSend}
            disabled={!value.trim() || disabled}
            className="mb-1 flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-lg bg-primary-600 text-white transition-all hover:bg-primary-700 disabled:opacity-40 disabled:cursor-not-allowed"
            title="Gửi (Enter)"
          >
            <Send className="h-4 w-4" />
          </button>
        </div>
        <ListeningIndicator visible={speech.isRecording || speech.isTranscribing} label={speech.statusLabel} />
        <p className="mt-1.5 text-center text-xs text-gray-400 dark:text-gray-600">
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
  const messagesEndRef = useRef(null);
  const lastSpokenAssistantIdRef = useRef(null);

  const AUTO_SEND_STT = String(import.meta.env.VITE_STT_AUTO_SEND || "false").toLowerCase() === "true";

  const handleSpeechError = useCallback((message) => {
    setError(message || "Đã xảy ra lỗi xử lý giọng nói. Vui lòng thử lại.");
  }, []);

  const tts = useTextToSpeech({ onError: handleSpeechError });

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
      const res = await chatAPI.ask(currentThreadId, content);
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
  }, [input, sending, threadId, onThreadCreated]);

  const speech = useSpeechRecognition({
    onTranscript: (text) => {
      setInput(text);
      if (AUTO_SEND_STT) {
        handleSend(text);
      }
    },
    onError: handleSpeechError,
  });

  const handleToggleSpeech = useCallback(async () => {
    if (!speech.isRecording && !speech.isTranscribing) {
      const prompted = sessionStorage.getItem("stt_prompted_this_session") === "1";
      if (!prompted) {
        const accepted = window.confirm(
          "Cho phép website truy cập microphone để nhập liệu bằng giọng nói?"
        );
        sessionStorage.setItem("stt_prompted_this_session", "1");
        if (!accepted) return;

        const granted = await speech.requestPermission();
        if (!granted) return;
      }
    }

    speech.toggleRecording();
  }, [speech]);

  const handleToggleTts = useCallback(() => {
    if (tts.isPlaying) {
      tts.stop();
      return;
    }

    const prompted = sessionStorage.getItem("tts_prompted_this_session") === "1";
    if (!prompted) {
      const accepted = window.confirm(
        "Cho phép website đọc phản hồi bằng giọng nói (TTS)?"
      );
      sessionStorage.setItem("tts_prompted_this_session", "1");
      if (!accepted) return;
    }

    tts.toggleEnabled();
  }, [tts]);

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
  }, [messages, sending]);

  useEffect(() => {
    const latestAssistant = [...messages].reverse().find((msg) => msg.role === "assistant");
    if (!latestAssistant || !tts.enabled || sending) return;
    if (lastSpokenAssistantIdRef.current === latestAssistant.id) return;

    lastSpokenAssistantIdRef.current = latestAssistant.id;
    tts.speak(latestAssistant.content);
  }, [messages, sending, tts]);

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
          speech={speech}
          tts={tts}
          onToggleSpeech={handleToggleSpeech}
          onToggleTts={handleToggleTts}
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
            <Heart className="h-10 w-10 text-primary-600 dark:text-primary-400" />
          </div>
          <h2 className="mb-2 text-2xl font-bold text-gray-900 dark:text-gray-100">
            Xin chào! 👋
          </h2>
          <p className="mb-8 max-w-md text-center text-gray-500 dark:text-gray-400">
            Tôi là Meddy — trợ lý y khoa AI. Hãy hỏi tôi về các vấn đề sức khỏe,
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
                className="flex items-start gap-3 rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 px-4 py-3 text-left text-sm text-gray-600 dark:text-gray-300 transition-all hover:border-primary-200 dark:hover:border-primary-700 hover:bg-primary-50 dark:hover:bg-primary-900/20 hover:shadow-sm"
              >
                <item.icon className="mt-0.5 h-4 w-4 flex-shrink-0 text-primary-500" />
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
          speech={speech}
          tts={tts}
          onToggleSpeech={handleToggleSpeech}
          onToggleTts={handleToggleTts}
        />
      </div>
    );
  }

  /* ── Active chat ─────────────────────────────────────────── */
  return (
    <div className="flex h-full flex-col bg-white dark:bg-gray-900">
      {/* Message list */}
      <div className="flex-1 overflow-y-auto divide-y divide-gray-100 dark:divide-gray-800">
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

      <ErrorBanner message={error} onDismiss={() => setError(null)} />
      <InputArea
        value={input}
        onChange={setInput}
        onSend={handleSend}
        disabled={sending}
        speech={speech}
        tts={tts}
        onToggleSpeech={handleToggleSpeech}
        onToggleTts={handleToggleTts}
      />
    </div>
  );
}

