// The doc-comment content gate over the TypeScript/JS surface — the TS half
// of tools/lints/docstring_slop.py + comment_slop.py. Reached through
// `checks.mjs doc-comments`; never run directly.
//
//   DS1  a /** */ block longer than 6 content lines
//   DS5  a /** */ block that only restates the declaration below it
//   DS6  an @param whose description only restates the parameter name
//   CM1  more than 3 consecutive own-line // comments
//   CM2  a banner (// ----, // ====) or // Step N inside a function body
//
// Pragmas (biome-ignore, eslint-, @ts-, prettier-) never count, and a file's
// leading header block is exempt: scripts/ci/CLAUDE.md requires it. Module,
// class and object-literal bodies may carry banners — a registry or a sample
// catalogue is supposed to be sectioned — so CM2 needs real function spans,
// which come from @babel/parser (already the TS parser behind bots-facts).
import { readFileSync } from "node:fs";
import { parse } from "@babel/parser";
import traverse from "@babel/traverse";

const BLOCK_MAX_LINES = 6;
const RUN_MAX = 3;

const JSDOC = /\/\*\*([\s\S]*?)\*\//g;
const PARAM = /^@param\s+(?:\{[^}]*\}\s*)?(\w+)\s*-?\s*(.*)$/;
const BANNER = /^\/\/\s*([-=#*~_]{3,}|step\s*\d+\b)/i;
const PRAGMA = /^\/\/\s*(biome-ignore|eslint|@ts-|prettier-|NOSONAR)/;
const FILLER = new Set(
  "the a an of to for and or in on is this that with from if by its it whether value values object id name current given".split(" "),
);

const words = (text) =>
  new Set(
    text
      .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
      .toLowerCase()
      .split(/[^a-z0-9]+/)
      .filter((w) => w && !FILLER.has(w)),
  );
const subset = (a, b) => a.size > 0 && [...a].every((w) => b.has(w));

const lineOf = (src, offset) => src.slice(0, offset).split("\n").length;

/** The first code line after `line` (1-based), skipping blanks and comments. */
function declarationBelow(lines, line) {
  for (let i = line; i < lines.length; i++) {
    const text = lines[i].trim();
    if (text && !text.startsWith("//") && !text.startsWith("*") && !text.startsWith("/*")) return text;
  }
  return "";
}

function checkBlocks(src, lines, report) {
  let first = true;
  for (const match of src.matchAll(JSDOC)) {
    const line = lineOf(src, match.index);
    const body = match[1]
      .split("\n")
      .map((l) => l.replace(/^\s*\*\s?/, "").trimEnd())
      .filter((l) => l.trim());
    // A leading block followed by a blank line is the file header; one that
    // touches the declaration below it documents that declaration.
    const isHeader = first && src.slice(0, match.index).trim() === "" && /^\n\s*\n/.test(src.slice(match.index + match[0].length));
    first = false;
    if (isHeader) continue;
    if (body.length > BLOCK_MAX_LINES) {
      report(line, "DS1", `JSDoc block is ${body.length} lines (max ${BLOCK_MAX_LINES}) — say what, not why`);
    }
    if (body.length === 1 && subset(words(body[0]), words(declarationBelow(lines, line + match[0].split("\n").length - 1)))) {
      report(line, "DS5", "JSDoc only restates the declaration below it — delete it");
    }
    for (const text of body) {
      const param = PARAM.exec(text.trim());
      if (param && subset(words(param[2]), words(param[1]))) {
        report(line, "DS6", `@param ${param[1]} only restates its name — drop it`);
      }
    }
  }
}

/** `[start, end]` line spans of every function body in the file. */
function functionSpans(src, path) {
  const ast = parse(src, {
    sourceType: "module",
    plugins: ["typescript", "jsx", "decorators"],
    errorRecovery: true,
    sourceFilename: path,
  });
  const spans = [];
  traverse.default(ast, {
    Function(nodePath) {
      const { loc } = nodePath.node.body;
      if (loc) spans.push([loc.start.line, loc.end.line]);
    },
  });
  return spans;
}

function checkLineComments(lines, spans, report) {
  let run = 0;
  let prev = -2;
  let seenCode = false;
  lines.forEach((raw, i) => {
    const text = raw.trim();
    const line = i + 1;
    if (!text.startsWith("//")) {
      if (text && !text.startsWith("*") && !text.startsWith("/*")) seenCode = true;
      return;
    }
    if (PRAGMA.test(text)) return;
    run = line === prev + 1 ? run + 1 : 1;
    prev = line;
    if (!seenCode) return; // the file header
    if (run === RUN_MAX + 1) {
      report(line - RUN_MAX, "CM1", `comment block longer than ${RUN_MAX} lines — keep the fact, drop the story`);
    }
    if (BANNER.test(text) && spans.some(([start, end]) => start < line && line < end)) {
      report(line, "CM2", "banner or step comment inside a function — split it instead of sectioning it");
    }
  });
}

export function checkDocComments(path) {
  const src = readFileSync(path, "utf8");
  const lines = src.split("\n");
  const findings = [];
  const report = (line, code, message) => findings.push({ path, line, code, message });
  checkBlocks(src, lines, report);
  checkLineComments(lines, functionSpans(src, path), report);
  return findings;
}
