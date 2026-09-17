"use client";

import type { CSSProperties } from "react";
import type { ReceiptPrinterMachineProps } from "@/features/pricing/components/receipt-printer.types";
import { cn } from "@/lib/utils";

/* The machine is always the dark charcoal unit with the black LCD, in both
   themes — only its backdrop changes. Tailwind scans source text and cannot
   see template-literal interpolation inside arbitrary values, so the hex
   tones are inlined below rather than referenced via constants. */

const machineClassName =
  "relative isolate w-full overflow-hidden bg-zinc-900 pb-8 [--printer-inner-radius:calc(var(--printer-radius)_-_var(--printer-inset))] [--printer-inset:0.75rem] [--printer-radius:1.5rem]";

const machineStyle: CSSProperties = {
  borderRadius: "var(--printer-radius)",
  padding: "var(--printer-inset)",
  paddingBottom: "2rem",
  boxShadow:
    "0 20px 36px -20px color-mix(in oklab,#18181b 55%,transparent),0 6px 14px -8px color-mix(in oklab,#18181b 24%,transparent),inset 0 1px 0 color-mix(in oklab,#fafafa 10%,transparent),inset 0 -1px 0 color-mix(in oklab,#18181b 55%,transparent)",
};

const noiseOverlayStyle: CSSProperties = {
  borderRadius: "inherit",
  backgroundImage: "url('/textures/plastic-noise.svg')",
  backgroundSize: "180px 180px",
};

export function ReceiptPrinterMachine({
  children,
  className,
  style,
  ...props
}: ReceiptPrinterMachineProps) {
  return (
    <div
      className={cn(machineClassName, className)}
      style={{ ...machineStyle, ...style }}
      {...props}
    >
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-0 z-0 bg-repeat opacity-30 mix-blend-multiply"
        style={noiseOverlayStyle}
      />
      {children}
      <div
        aria-hidden="true"
        className="absolute inset-x-6 bottom-[var(--printer-inset)] z-40 h-2 rounded-sm bg-zinc-950 shadow-inner shadow-zinc-950"
      />
    </div>
  );
}
