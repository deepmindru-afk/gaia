/**
 * Splash Window Module
 *
 * Creates and manages the splash screen shown immediately on app
 * launch while the Next.js server and main window initialise in
 * the background.
 *
 * The splash renders a **dark app-shell skeleton** (sidebar + main +
 * composer, see splash.html) in a window filling the primary display's
 * work area — the same footprint the main window maximises into — with the
 * same `hiddenInset` chrome and dark `backgroundColor`. So the boot reads as
 * "GAIA opened and is loading its content", and the splash → maximised main
 * swap is a seamless no-flash, no-jump transition. It uses `show: true` so it
 * appears instantly — no waiting for `dom-ready`.
 *
 * @module windows/splash
 */

import { join } from "node:path";
import { app, BrowserWindow, screen } from "electron";

/** Solid backdrop under the skeleton — matches the main window's
 * `backgroundColor` so revealing the main window is a no-flash swap. */
const SPLASH_BACKGROUND = "#000000";

/** Reference to the current splash window (if any). */
let splashWindow: BrowserWindow | null = null;

/**
 * Create and display the splash screen.
 *
 * This **must** be the very first visual operation in the
 * startup flow — no blocking code should run before it.
 *
 * Fills the primary display's work area (the footprint the main window
 * maximises into) with `hiddenInset` chrome and a dark background, so the
 * skeleton covers the screen exactly like the maximised app and the swap has
 * no jump or reposition.
 */
export function createSplashWindow(): void {
  const { workArea } = screen.getPrimaryDisplay();

  splashWindow = new BrowserWindow({
    x: workArea.x,
    y: workArea.y,
    width: workArea.width,
    height: workArea.height,
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
