"use client";

import { useEffect, useRef } from "react";

import { ANALYTICS_EVENTS, trackEvent } from "@/lib/analytics";

import type { OnboardingState } from "../state/types";

export function useOnboardingAnalytics(state: OnboardingState): void {
  const startedRef = useRef(false);

  useEffect(() => {
    if (startedRef.current) return;
    startedRef.current = true;
    trackEvent(ANALYTICS_EVENTS.ONBOARDING_STARTED, {
      has_saved_state:
        state.questionIndex > 0 || Object.keys(state.responses).length > 0,
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Step and completion analytics live server-side: POST /onboarding/phase
  // emits onboarding:step_completed and the worker emits
  // onboarding:completed on PERSONALIZATION_COMPLETE. Emitting the same names
  // here would double-count every step and completion.
}
