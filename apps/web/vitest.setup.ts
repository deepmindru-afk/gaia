import { vi } from "vitest";

// HeroUI's ripple loads framer-motion features lazily and sets state after a
// file's jsdom teardown ("window is not defined"); it is chrome, not behaviour.
vi.mock("@heroui/ripple", () => ({
  Ripple: () => null,
  useRipple: () => ({ ripples: [], onPress: vi.fn(), onClear: vi.fn() }),
}));
