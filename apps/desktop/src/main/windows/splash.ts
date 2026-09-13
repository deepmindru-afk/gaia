/**
 * Splash Window Module
 *
 * Creates and manages the splash screen shown immediately on app
 * launch while the Next.js server and main window initialise in
 * the background.
 *
 * The splash renders a **dark app-shell skeleton** (sidebar + main +
 * composer, see splash.html) in a **small centered window** with the
 * same `hiddenInset` chrome and dark `backgroundColor` as the main
 * window. When the app is ready the main window opens at the same
 * small bounds and then maximises — so the boot reads as a loader
 * that scales up into the full app. It uses `show: true` so it
 * appears instantly — no waiting for `dom-ready`.
 *
 * @module windows/splash
 */

import { join } from "node:path";
import { app, BrowserWindow, screen } from "electron";

/** Solid backdrop under the skeleton — matches the main window's
 * `backgroundColor` so revealing the main window is a no-flash swap. */
const SPLASH_BACKGROUND = "#000000";

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

/**
 * Resolve the centered loader bounds for the primary display.
 *
 * @returns `{ x, y, width, height }` in screen coordinates — a loader-sized
 *   window centered in the work area.
 */
export function getLoaderBounds(): {
  x: number;
  y: number;
  width: number;
  height: number;
} {
  const { workArea } = screen.getPrimaryDisplay();
  const { width, height } = resolveLoaderSize(workArea);
  const x = Math.round(workArea.x + Math.max(0, (workArea.width - width) / 2));
  const y = Math.round(
    workArea.y + Math.max(0, (workArea.height - height) / 2),
  );
  return { x, y, width, height };
}

/** Reference to the current splash window (if any). */
let splashWindow: BrowserWindow | null = null;

/**
 * Create and display the splash screen.
 *
 * This **must** be the very first visual operation in the
 * startup flow — no blocking code should run before it.
 *
 * Opens a centered loader (see {@link getLoaderBounds}) with `hiddenInset`
 * chrome and a dark background. The main window opens at these same bounds
 * and then maximises (see windows/main.ts), so the boot reads as the loader
 * scaling up into the full app.
 */
export function createSplashWindow(): void {
  const { x, y, width, height } = getLoaderBounds();

  splashWindow = new BrowserWindow({
    x,
    y,
    width,
    height,
    backgroundColor: SPLASH_BACKGROUND,
    // Match the main window's chrome (windows/main.ts) so the skeleton looks
    // like the real app window (traffic lights) and the swap is seamless.
    titleBarStyle: "hiddenInset",
    trafficLightPosition: { x: 16, y: 16 },
    resizable: false,
    minimizable: false,
    maximizable: false,
    alwaysOnTop: false,
    skipTaskbar: false,
    focusable: true,
    show: true,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
    },
  });

  const splashPath = app.isPackaged
    ? join(process.resourcesPath, "splash.html")
    : join(__dirname, "../../resources/splash.html");

  splashWindow.loadFile(splashPath);
}

/**
 * Destroy the splash window and release its reference.
 *
 * Uses `destroy()` rather than `close()` to guarantee the
 * window is removed immediately without firing close events.
 */
export function closeSplashWindow(): void {
  if (splashWindow && !splashWindow.isDestroyed()) {
    splashWindow.destroy();
    splashWindow = null;
    console.log("[Main] Splash window destroyed");
  }
}

/**
 * Check whether the splash window still exists and is not destroyed.
 *
 * @returns `true` if the splash is still visible.
 */
export function isSplashAlive(): boolean {
  return splashWindow !== null && !splashWindow.isDestroyed();
}
