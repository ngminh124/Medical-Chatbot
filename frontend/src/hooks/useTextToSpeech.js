import { useCallback, useEffect, useRef, useState } from "react";
import { speechAPI } from "../api/speech";

const DEFAULT_VOICE_ID = import.meta.env.VITE_TTS_VOICE_ID || "";

export function useTextToSpeech({ onError } = {}) {
  const [enabled, setEnabled] = useState(false);
  const [isAvailable, setIsAvailable] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);

  const audioRef = useRef(null);
  const abortRef = useRef(null);
  const cacheRef = useRef(new Map());

  const refreshHealth = useCallback(async () => {
    try {
      await speechAPI.ttsHealth();
      setIsAvailable(true);
      return true;
    } catch (err) {
      const status = err?.response?.status;
      setIsAvailable(status === 503 ? false : true);
      return status !== 503;
    }
  }, []);

  const stop = useCallback(() => {
    if (!audioRef.current) return;
    audioRef.current.pause();
    audioRef.current.currentTime = 0;
    setIsPlaying(false);
  }, []);

  const playAudio = useCallback(async (audioUrl) => {
    let audio = audioRef.current;

    if (!audio) {
      audio = new Audio();
      audioRef.current = audio;
    }

    stop();

    audio.src = audioUrl;
    audio.onended = () => setIsPlaying(false);
    audio.onerror = () => {
      setIsPlaying(false);
      onError?.("Không thể phát âm thanh. Vui lòng thử lại.");
    };

    try {
      setIsPlaying(true);
      await audio.play();
    } catch {
      setIsPlaying(false);
      onError?.("Trình duyệt đã chặn phát âm thanh tự động. Vui lòng bấm phát thủ công.");
    }
  }, [onError, stop]);

  const speak = useCallback(
    async (text, voiceId = DEFAULT_VOICE_ID) => {
      const normalizedText = text?.trim();
      if (!enabled || !normalizedText) return;

      if (!isAvailable) {
        const ok = await refreshHealth();
        if (!ok) {
          onError?.("TTS chưa sẵn sàng");
          return;
        }
      }

      abortRef.current?.abort();

      const cacheKey = `${voiceId || "default"}::${normalizedText}`;
      const cachedUrl = cacheRef.current.get(cacheKey);
      if (cachedUrl) {
        await playAudio(cachedUrl);
        return;
      }

      const controller = new AbortController();
      abortRef.current = controller;

      setIsLoading(true);
      try {
        const res = await speechAPI.synthesize(
          { text: normalizedText, voice_id: voiceId || undefined },
          { signal: controller.signal }
        );

        const audioBlob = res?.data;
        if (!(audioBlob instanceof Blob) || audioBlob.size === 0) {
          throw new Error("Invalid audio response");
        }

        const audioUrl = URL.createObjectURL(audioBlob);
        cacheRef.current.set(cacheKey, audioUrl);
        await playAudio(audioUrl);
      } catch (err) {
        const status = err?.response?.status;
        if (status === 503) {
          setIsAvailable(false);
          setEnabled(false);
          onError?.("TTS chưa sẵn sàng");
        } else if (err?.name !== "CanceledError" && err?.code !== "ERR_CANCELED") {
          onError?.("Không thể tổng hợp giọng nói. Vui lòng thử lại.");
        }
      } finally {
        setIsLoading(false);
        abortRef.current = null;
      }
    },
    [enabled, isAvailable, onError, playAudio, refreshHealth]
  );

  const toggleEnabled = useCallback(() => {
    const run = async () => {
      if (!isAvailable) {
        const ok = await refreshHealth();
        if (!ok) {
          onError?.("TTS chưa sẵn sàng");
          return;
        }
      }

      setEnabled((prev) => {
        const next = !prev;
        if (!next) stop();
        return next;
      });
    };

    run();
  }, [isAvailable, onError, refreshHealth, stop]);

  const enable = useCallback(() => {
    const run = async () => {
      if (!isAvailable) {
        const ok = await refreshHealth();
        if (!ok) {
          onError?.("TTS chưa sẵn sàng");
          return;
        }
      }
      setEnabled(true);
    };

    run();
  }, [isAvailable, onError, refreshHealth]);

  const disable = useCallback(() => {
    setEnabled(false);
    stop();
  }, [stop]);

  useEffect(() => {
    let cancelled = false;

    speechAPI
      .ttsHealth()
      .then(() => {
        if (!cancelled) setIsAvailable(true);
      })
      .catch((err) => {
        if (cancelled) return;
        setIsAvailable(err?.response?.status !== 503 ? true : false);
      });

    return () => {
      cancelled = true;
      abortRef.current?.abort();
      stop();

      cacheRef.current.forEach((url) => URL.revokeObjectURL(url));
      cacheRef.current.clear();

      if (audioRef.current) {
        audioRef.current.src = "";
        audioRef.current.onended = null;
        audioRef.current.onerror = null;
      }
    };
  }, [stop]);

  return {
    enabled,
    isAvailable,
    isLoading,
    isPlaying,
    speak,
    stop,
    enable,
    disable,
    refreshHealth,
    toggleEnabled,
  };
}
