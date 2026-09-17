"use client";

import type { CSSProperties } from "react";
import type { ReceiptPrinterScreenProps } from "@/features/pricing/components/receipt-printer.types";
import { cn } from "@/lib/utils";

const screenGlowStyle: CSSProperties = {
  borderRadius: "inherit",
  boxShadow: "inset 0 0 24px 4px color-mix(in oklab,#09090b 35%,transparent)",
};

export function ReceiptPrinterScreen({
  children,
  className,
  style,
  ...props
}: ReceiptPrinterScreenProps) {
  return (
    <div
      className={cn(
        "relative z-10 isolate overflow-hidden bg-zinc-800 p-4 text-zinc-50 shadow-inner shadow-zinc-950/30",
        className,
      )}
      style={{ borderRadius: "var(--printer-inner-radius)", ...style }}
      {...props}
    >
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-0 z-20"
        style={screenGlowStyle}
      />
      <div className="relative z-10">{children}</div>
    </div>
  );
}
