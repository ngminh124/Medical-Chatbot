import { useCallback, useEffect, useRef, useState } from "react";

/**
 * Generic sending hook with abort support.
 * @param {(payload: any, ctx: { signal: AbortSignal }) => Promise<void>} sender
 */
export function useSendMessage(sender) {
  const [isSending, setIsSending] = useState(false);
  const abortControllerRef = useRef(null);

  const stopSending = useCallback(() => {
    abortControllerRef.current?.abort();
    abortControllerRef.current = null;
    setIsSending(false);
  }, []);

  const sendMessageStream = useCallback(
    async (payload) => {
      if (isSending) return;

      const controller = new AbortController();
      abortControllerRef.current = controller;
      setIsSending(true);

      try {
        await sender(payload, { signal: controller.signal });
      } finally {
        if (abortControllerRef.current === controller) {
          abortControllerRef.current = null;
        }
        setIsSending(false);
      }
    },
    [isSending, sender]
  );

  const sendMessage = sendMessageStream;

  useEffect(() => {
    return () => {
      abortControllerRef.current?.abort();
      abortControllerRef.current = null;
    };
  }, []);

  return {
    isSending,
    sendMessage,
    sendMessageStream,
    stopSending,
    abortControllerRef,
  };
}
