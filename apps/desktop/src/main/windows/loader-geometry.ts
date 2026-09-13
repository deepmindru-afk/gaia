/**
 * Loader Geometry Module
 *
 * Pure window-size math for the boot loader (splash + main window start
 * bounds). Dependency-free on purpose: anything importing Electron cannot be
 * unit-tested on machines without the Electron binary (e.g. Linux CI), so the
 * geometry lives here and the Electron calls stay in `windows/splash`.
 *
 * @module windows/loader-geometry
 */

/** Loader size as a fraction of the display work area — big enough that the
 * shell skeleton is legible, small enough to still read as a loader that
 * scales up into the full app. */
const LOADER_WORK_AREA_RATIO = 0.72;

/** Bounds for the loader so it stays usable on tiny and huge displays. */
const LOADER_MIN_WIDTH = 880;
const LOADER_MIN_HEIGHT = 600;
const LOADER_MAX_WIDTH = 1280;
const LOADER_MAX_HEIGHT = 820;

/** Work-area size (display minus menu bar / Dock). */
export interface WorkAreaSize {
  width: number;
  height: number;
}

/**
 * Resolve the loader window size for a given work area.
 *
 * Pure (no Electron dependency) so it is unit-testable: 72% of the work area,
 * clamped to the usable bounds, and never larger than the work area itself —
 * so the loader fits on every screen from a tiny 800×600 panel to a 4K
 * display.
 *
 * @param workArea - The display work-area size in CSS pixels.
 * @returns `{ width, height }` in CSS pixels, guaranteed to fit.
 */
export function resolveLoaderSize(workArea: WorkAreaSize): {
  width: number;
  height: number;
} {
  const width = Math.round(
    Math.min(
      Math.max(workArea.width * LOADER_WORK_AREA_RATIO, LOADER_MIN_WIDTH),
      LOADER_MAX_WIDTH,
      workArea.width,
    ),
  );
  const height = Math.round(
    Math.min(
      Math.max(workArea.height * LOADER_WORK_AREA_RATIO, LOADER_MIN_HEIGHT),
      LOADER_MAX_HEIGHT,
      workArea.height,
    ),
  );
  return { width, height };
}
