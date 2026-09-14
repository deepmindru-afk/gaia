"use client";

import { useCallback, useState } from "react";
import { toast } from "@/lib/toast";

import { pricingApi } from "../api/pricingApi";
import { LAST_CHECKOUT_PRODUCT_KEY } from "../constants";

export const useDodoPayments = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const createSubscriptionAndRedirect = useCallback(
    async (productId: string, discountCode?: string) => {
      setIsLoading(true);
      setError(null);

      try {
        const result = await pricingApi.createSubscription({
          product_id: productId,
          ...(discountCode ? { discount_code: discountCode } : {}),
        });

        // Redirect user to Dodo payment link
        if (result.payment_link) {
          // Remember the plan so the result page can restart checkout on retry.
          localStorage.setItem(LAST_CHECKOUT_PRODUCT_KEY, productId);
          window.location.href = result.payment_link;
        } else {
          throw new Error("Payment link not received");
        }
      } catch (err) {
        const errorMessage =
          err instanceof Error ? err.message : "Failed to create subscription";
        setError(errorMessage);
        toast.error(errorMessage);
      } finally {
        setIsLoading(false);
      }
    },
    [],
  );

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  return {
    createSubscriptionAndRedirect,
    isLoading,
    error,
    clearError,
  };
};
