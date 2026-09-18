import { Button } from "@heroui/button";
import {
  Dropdown,
  DropdownItem,
  DropdownMenu,
  DropdownTrigger,
} from "@heroui/dropdown";
import { Input } from "@heroui/input";
import { MoreHorizontalIcon } from "@icons";
import type {
  ApprovalDecision,
  ApprovalRequestData,
  ApprovalScope,
  ApprovalStatus,
} from "@shared/chat";
import { formatApprovalAge, RECONFIRM_AGE_SECONDS } from "@shared/chat";
import { useEffect, useRef, useState } from "react";
import { ShieldAlertIcon } from "@/components/shared/icons";
import { chatApi } from "@/features/chat/api/chatApi";
import { useMarkApprovalDecided } from "@/features/chat/hooks/useMarkApprovalDecided";
import { formatToolName } from "@/features/chat/utils/chatUtils";
import { toast } from "@/lib/toast";

interface ApprovalRequestSectionProps {
  data: ApprovalRequestData;
  onDecided: (status: ApprovalStatus, feedback: string | null) => void;
  /** A batch decision ("Approve all"/"Decline all") is in flight — lock this card
   * so a per-card click can't send a second, conflicting decision for the same id. */
  disabled?: boolean;
}

function ArgsPreview({ args }: { args: Record<string, unknown> }) {
  const rows = Object.entries(args).filter(
    ([, value]) =>
      typeof value === "string" ||
      typeof value === "number" ||
      typeof value === "boolean",
  );
  if (rows.length === 0) return null;
  return (
    <div className="mt-3 space-y-2 rounded-2xl bg-zinc-900 p-3">
      {rows.map(([key, value]) => (
        <div key={key} className="text-xs">
          <div className="mb-0.5 text-[11px] text-zinc-500">
            {key.replaceAll("_", " ")}
          </div>
          <div className="text-zinc-200">{String(value)}</div>
        </div>
      ))}
    </div>
  );
}

/** Pre-commit pause on approve taps: the decision hasn't touched the ledger
 * yet, so regret costs nothing. A client constant, never ledger state. */
const COMMIT_GRACE_MS = 4000;

type Phase = "idle" | "reconfirm" | "grace" | "submitting";

export default function ApprovalRequestSection({
  data,
  onDecided,
  disabled = false,
}: ApprovalRequestSectionProps) {
  const [submitting, setSubmitting] = useState<ApprovalDecision | null>(null);
  const [feedback, setFeedback] = useState("");
  const [phase, setPhase] = useState<Phase>("idle");
  // A stale-v tap committed nothing; the next submit omits v so the CAS —
  // not the version check — decides. v is an optimization, never a gate.
  const [versionConflict, setVersionConflict] = useState(false);
  const graceTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const locked = submitting !== null || disabled || phase === "submitting";
  const markApprovalDecided = useMarkApprovalDecided();

  useEffect(
    () => () => {
      if (graceTimer.current !== null) clearTimeout(graceTimer.current);
    },
    [],
  );

  const needsReconfirm = (data.age_seconds ?? 0) >= RECONFIRM_AGE_SECONDS;

  const submit = async (
    decision: ApprovalDecision,
    scope: ApprovalScope = "once",
  ) => {
    setSubmitting(decision);
    setPhase("submitting");
    try {
      const outcome = await chatApi.postApprovalDecision(data.approval_id, {
        decision,
        feedback: feedback.trim() || undefined,
        scope,
        v: versionConflict ? undefined : (data.ledger_version ?? undefined),
      });
      if (!outcome.success) {
        // Stale tap: the row moved under this card. Settle locally when the
        // row is already decided; otherwise keep the card and drop the
        // version so the next tap goes through the CAS directly.
        if (outcome.status === "approved" || outcome.status === "denied") {
          markApprovalDecided();
          onDecided(
            outcome.status === "approved" ? "approved" : "denied",
            feedback.trim() || null,
          );
        } else {
          setVersionConflict(true);
          setSubmitting(null);
          setPhase("idle");
          toast.error("That approval already moved — tap again to confirm");
        }
        return;
      }
      // Settle locally: the resolved frame is published on the RESUMED run's
      // stream (a different message), so it never replaces this card. A 410
      // (already resolved elsewhere) is swallowed by postApprovalDecision and
      // settles here too; reaching the catch means the submit genuinely failed.
      markApprovalDecided();
      onDecided(
        decision === "approve" ? "approved" : "denied",
        feedback.trim() || null,
      );
    } catch {
      toast.error("Couldn't submit your decision — please try again");
      setSubmitting(null);
      setPhase("idle");
    }
  };

  const onApproveTap = () => {
    if (needsReconfirm && phase === "idle") {
      setPhase("reconfirm");
      return;
    }
    setPhase("grace");
    graceTimer.current = setTimeout(() => submit("approve"), COMMIT_GRACE_MS);
  };

  const cancelGrace = () => {
    if (graceTimer.current !== null) {
      clearTimeout(graceTimer.current);
      graceTimer.current = null;
    }
    setPhase("idle");
  };

  if (data.status !== "pending") return null;

  if (phase === "reconfirm") {
    return (
      <div className="w-full max-w-md rounded-2xl bg-zinc-800 p-4 text-white">
        <div className="text-sm leading-snug text-zinc-100">
          {`Asked ${formatApprovalAge(data.age_seconds).replace("asked ", "")} ago — still want this?`}
        </div>
        <div className="mt-1 text-xs text-zinc-400">{data.summary}</div>
        <div className="mt-3 flex items-center gap-2">
          <Button
            color="primary"
            size="sm"
            onPress={() => {
              setPhase("idle");
              onApproveTap();
            }}
          >
            Yes, still approve
          </Button>
          <Button variant="flat" size="sm" onPress={() => setPhase("idle")}>
            Back
          </Button>
        </div>
      </div>
    );
  }

  if (phase === "grace") {
    return (
      <div className="w-full max-w-md rounded-2xl bg-zinc-800 p-4 text-white">
        <div className="text-sm leading-snug text-zinc-100">
          Sending your approval…
        </div>
        <div className="mt-3 flex items-center gap-2">
          <Button variant="flat" size="sm" onPress={cancelGrace}>
            Cancel
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full max-w-md rounded-2xl bg-zinc-800 p-4 text-white">
      <div className="flex items-start gap-2.5">
        <div className="flex size-8 shrink-0 items-center justify-center rounded-xl bg-amber-400/10">
          <ShieldAlertIcon width={17} height={17} className="text-amber-400" />
        </div>
        <div className="min-w-0 flex-1">
          <div className="text-xs font-medium text-amber-400">
            Needs approval
          </div>
          <div className="text-sm leading-snug text-zinc-100">
            {formatToolName(data.gated_tool_name)}
          </div>
          {data.age_seconds != null && (
            <div className="mt-0.5 text-[11px] text-zinc-500">
              {formatApprovalAge(data.age_seconds)}
            </div>
          )}
        </div>
      </div>

      {data.rationale?.trim() && (
        <div className="mt-2 text-xs leading-snug text-zinc-400">
          {data.rationale.trim()}
        </div>
      )}

      <ArgsPreview args={data.args_preview} />

      <div className="mt-3 flex items-center gap-2">
        <Input
          className="flex-1"
          size="sm"
          variant="flat"
          placeholder="Tell GAIA why (optional)"
          value={feedback}
          onValueChange={setFeedback}
          isDisabled={locked}
        />
        <Button
          color="primary"
          size="sm"
          isLoading={submitting === "approve"}
          isDisabled={locked}
          onPress={onApproveTap}
        >
          Approve
        </Button>
        <Button
          variant="flat"
          size="sm"
          isLoading={submitting === "deny"}
          isDisabled={locked}
          onPress={() => submit("deny")}
        >
          Deny
        </Button>
        <Dropdown placement="bottom-end">
          <DropdownTrigger>
            <Button
              isIconOnly
              size="sm"
              variant="light"
              isDisabled={locked}
              aria-label="More approval options"
            >
              <MoreHorizontalIcon width={18} />
            </Button>
          </DropdownTrigger>
          <DropdownMenu aria-label="Approval options">
            <DropdownItem
              key="always"
              onPress={() => submit("approve", "always_tool")}
            >
              Always allow this tool
            </DropdownItem>
          </DropdownMenu>
        </Dropdown>
      </div>
    </div>
  );
}
