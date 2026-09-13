/**
 * electron-builder afterSign hook: ad-hoc-signs local builds (real signing
 * disabled via CSC_IDENTITY_AUTO_DISCOVERY=false leaves only Electron's broken
 * linker seal). macOS 26's RunningBoard kills such apps ~12s after launch (Dock
 * icon appears then vanishes); this restores a valid seal so they run from Finder.
 */
const { execFileSync } = require("node:child_process");
const path = require("node:path");

module.exports = async function adhocSign(context) {
  if (context.electronPlatformName !== "darwin") return;
  if (process.env.CSC_IDENTITY_AUTO_DISCOVERY !== "false") return;

  const appName = `${context.packager.appInfo.productFilename}.app`;
  const appPath = path.join(context.appOutDir, appName);

  console.log(`  • ad-hoc signing (real signing disabled)  file=${appPath}`);
  // Absolute path to the SIP-protected system binary — never resolve via
  // $PATH, which a caller could repoint at a malicious `codesign`.
  execFileSync("/usr/bin/codesign", ["--force", "--deep", "--sign", "-", appPath], {
    stdio: "inherit",
  });
};
