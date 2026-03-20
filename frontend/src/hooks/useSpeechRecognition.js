import { useCallback, useEffect, useRef, useState } from "react";
import { speechAPI } from "../api/speech";

const MAX_RECORDING_MS = Number(import.meta.env.VITE_STT_MAX_RECORDING_MS || 20000);

function getSupportedMimeType() {
  if (typeof MediaRecorder === "undefined") return "";

  const types = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/ogg"];
  return types.find((type) => MediaRecorder.isTypeSupported(type)) || "";
}

function getFileNameFromMime(mimeType) {
  if (mimeType.includes("ogg")) return "recording.ogg";
  return "recording.webm";
}

export function useSpeechRecognition({ onTranscript, onError } = {}) {
  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [isAvailable, setIsAvailable] = useState(true);
  const [permissionState, setPermissionState] = useState("prompt");
  const [statusLabel, setStatusLabel] = useState("");

  const streamRef = useRef(null);
  const recorderRef = useRef(null);
  const chunksRef = useRef([]);
  const timeoutRef = useRef(null);
  const abortRef = useRef(null);

  const cleanupStream = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
  }, []);

  const clearRecordingTimeout = useCallback(() => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }
  }, []);

  const requestPermission = useCallback(async () => {
    if (!navigator?.mediaDevices?.getUserMedia) {
      onError?.("Trình duyệt hiện tại chưa hỗ trợ microphone.");
      return false;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      stream.getTracks().forEach((track) => track.stop());
      setPermissionState("granted");
      return true;
    } catch (err) {
      const denied =
        err?.name === "NotAllowedError" ||
        err?.name === "PermissionDeniedError" ||
        err?.message?.toLowerCase().includes("permission");

      setPermissionState(denied ? "denied" : "prompt");
      onError?.(
        denied
          ? "Bạn đã từ chối quyền microphone. Vui lòng cấp quyền trong trình duyệt để dùng nhập giọng nói."
          : "Không thể truy cập microphone. Vui lòng kiểm tra thiết bị và thử lại."
      );
      return false;
    }
  }, [onError]);

  const stopRecording = useCallback(() => {
    clearRecordingTimeout();
    const recorder = recorderRef.current;

    if (recorder && recorder.state !== "inactive") {
      recorder.stop();
    }
  }, [clearRecordingTimeout]);

  const transcribeBlob = useCallback(
    async (blob) => {
      setIsTranscribing(true);
      setStatusLabel("Đang xử lý giọng nói...");

      const formData = new FormData();
      formData.append("file", blob, getFileNameFromMime(blob.type || "audio/webm"));

      const controller = new AbortController();
      abortRef.current = controller;

      try {
        const res = await speechAPI.transcribe(formData, { signal: controller.signal });
        const text = res?.data?.text?.trim();
        if (text) {
          onTranscript?.(text, res.data);
        } else {
          onError?.("Không nhận diện được nội dung giọng nói. Vui lòng thử lại.");
        }
      } catch (err) {
        const status = err?.response?.status;
        if (status === 503) {
          setIsAvailable(false);
          onError?.("STT chưa sẵn sàng");
        } else if (err?.name !== "CanceledError" && err?.code !== "ERR_CANCELED") {
          onError?.("Không thể chuyển giọng nói thành văn bản. Vui lòng thử lại.");
        }
      } finally {
        setIsTranscribing(false);
        setStatusLabel("");
        abortRef.current = null;
      }
    },
    [onError, onTranscript]
  );

  const startRecording = useCallback(async () => {
    if (!navigator?.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
      onError?.("Trình duyệt hiện tại chưa hỗ trợ ghi âm.");
      return;
    }

    if (!isAvailable) {
      onError?.("STT chưa sẵn sàng");
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      setPermissionState("granted");

      const mimeType = getSupportedMimeType();
      const recorder = mimeType
        ? new MediaRecorder(stream, { mimeType })
        : new MediaRecorder(stream);

      recorderRef.current = recorder;
      chunksRef.current = [];

      recorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      recorder.onstop = async () => {
        setIsRecording(false);
        setStatusLabel("");
        clearRecordingTimeout();

        const finalBlob = new Blob(chunksRef.current, {
          type: recorder.mimeType || mimeType || "audio/webm",
        });
        chunksRef.current = [];
        cleanupStream();

        if (finalBlob.size > 0) {
          await transcribeBlob(finalBlob);
        }
      };

      recorder.start(250);
      setIsRecording(true);
      setStatusLabel("Đang nghe...");

      timeoutRef.current = setTimeout(() => {
        if (recorderRef.current?.state === "recording") {
          onError?.("Đã tự động dừng ghi âm do quá thời gian chờ.");
          stopRecording();
        }
      }, MAX_RECORDING_MS);
    } catch (err) {
      const denied =
        err?.name === "NotAllowedError" ||
        err?.name === "PermissionDeniedError" ||
        err?.message?.toLowerCase().includes("permission");

      setPermissionState(denied ? "denied" : "prompt");
      onError?.(
        denied
          ? "Bạn chưa cấp quyền microphone. Vui lòng cho phép truy cập micro và thử lại."
          : "Không thể truy cập microphone. Vui lòng kiểm tra thiết bị và thử lại."
      );
      cleanupStream();
      setIsRecording(false);
      setStatusLabel("");
    }
  }, [cleanupStream, isAvailable, onError, stopRecording, transcribeBlob]);

  const toggleRecording = useCallback(() => {
    if (isTranscribing) return;
    if (isRecording) {
      stopRecording();
      return;
    }
    startRecording();
  }, [isRecording, isTranscribing, startRecording, stopRecording]);

  useEffect(() => {
    let cancelled = false;

    if (navigator?.permissions?.query) {
      navigator.permissions
        .query({ name: "microphone" })
        .then((status) => {
          if (!cancelled) {
            setPermissionState(status.state);
          }
          status.onchange = () => {
            if (!cancelled) {
              setPermissionState(status.state);
            }
          };
        })
        .catch(() => {
          if (!cancelled) setPermissionState("prompt");
        });
    }

    speechAPI
      .sttHealth()
      .then(() => {
        if (!cancelled) setIsAvailable(true);
      })
      .catch((err) => {
        if (cancelled) return;
        setIsAvailable(err?.response?.status !== 503 ? true : false);
      });

    return () => {
      cancelled = true;
      clearRecordingTimeout();
      abortRef.current?.abort();
      stopRecording();
      cleanupStream();
    };
  }, [cleanupStream, clearRecordingTimeout, stopRecording]);

  return {
    isAvailable,
    isRecording,
    isTranscribing,
    permissionState,
    statusLabel,
    requestPermission,
    toggleRecording,
    stopRecording,
  };
}
