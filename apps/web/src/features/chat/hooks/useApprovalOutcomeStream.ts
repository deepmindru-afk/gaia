import { fetchEventSource } from "@microsoft/fetch-event-source";
import type { ApprovalStatus } from "@shared/chat";
import { useCallback, useEffect, useRef, useState } from "react";
import { desktopClientHeaders } from "@/lib/electron/api";

const TERMINAL_OUTCOMES: ReadonlySet<string> = new Set([
  "approved",
  "denied",
  "revoked",
  "executed",
  "failed",
  "unknown",
]);

export type OutcomeStreamPhase = "idle" | "running" | "done";

/**
 * Follow one approval tap to its outcome over SSE.
 *
 * The tap settles the card optimistically, which used to leave a silent gap:
 * no stream activity, no indicator, until some other frame happened to move.
 * This opens a short decision stream (`GET /approvals/{id}/events`) that says
 * `running` now and the terminal outcome when the ledger lands there, then
 * closes. Socket drop or timeout just ends the stream — the websocket
 * broadcast and reload truth still settle the card, so nothing can stick.
 */
export function useApprovalOutcomeStream() {
  const [phase, setPhase] = useState<OutcomeStreamPhase>("idle");
  const [outcome, setOutcome] = useState<ApprovalStatus | null>(null);
  const controller = useRef<AbortController | null>(null);

  const stop = useCallback(() => {
    controller.current?.abort();
    controller.current = null;
  }, []);

  useEffect(() => stop, [stop]);

  const start = useCallback(
    (approvalId: string) => {
      stop();
      setOutcome(null);
      setPhase("running");
      const aborter = new AbortController();
      controller.current = aborter;
      void fetchEventSource(
        `${process.env.NEXT_PUBLIC_API_BASE_URL}approvals/${approvalId}/events`,
        {
          method: "GET",
          openWhenHidden: true,
          headers: {
            Accept: "text/event-stream",
            ...desktopClientHeaders(),
          },
          credentials: "include",
          signal: aborter.signal,
          onmessage(event) {
            let status: string | null = null;
            try {
              status =
                (JSON.parse(event.data) as { status?: string }).status ?? null;
            } catch {
              return;
            }
            if (
              status === null ||
              status === "running" ||
              status === "executing"
            )
              return;
            if (TERMINAL_OUTCOMES.has(status)) {
              setOutcome(status as ApprovalStatus);
              setPhase("done");
              aborter.abort();
            }
          },
          onclose() {
            // Server closed without an outcome (timeout/race): end the row.
            // The websocket broadcast and reload truth still settle the card.
            setPhase("done");
          },
          onerror() {
            setPhase("done");
          },
        },
      );
    },
    [stop],
  );

  return { phase, outcome, start, stop };
}
