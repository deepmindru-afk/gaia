import { describe, expect, it } from "vitest";
import {
  couldBecomeReactDirective,
  reactDirectiveEmoji,
} from "./react-directive";

// Same table as REACT_DIRECTIVE_CASES in apps/api/tests/unit/agents/test_comms_directive.py —
// change both together, or bots and backend disagree on what gets an emoji_ack.
const REACT_DIRECTIVE_CASES: readonly (readonly [string, string | null])[] = [
  ["REACT: 👍", "👍"],
  ["  react:   ✅  ", "✅"],
  ["REACT:👍", "👍"],
  ["REACT: 👍\n", "👍"],
  ["REACT: 😎<NEW_MESSAGE_BREAK>", "😎"],
  ["REACT: <NEW_MESSAGE_BREAK>", null],
  ["REACT:", null],
  ["REACT:   ", null],
  ["REACT: 👍\nand more", null],
  ["REACTION: completed", null],
  ["hello REACT: 👍", null],
  ["Booked your 9am flight to Tokyo.", null],
];

describe("reactDirectiveEmoji", () => {
  it.each(REACT_DIRECTIVE_CASES)("classifies %j as %j", (text, emoji) => {
    expect(reactDirectiveEmoji(text)).toBe(emoji);
  });
});

describe("couldBecomeReactDirective", () => {
  it.each(REACT_DIRECTIVE_CASES.filter(([, emoji]) => emoji !== null))(
    "holds the finished directive %j",
    (text) => {
      expect(couldBecomeReactDirective(text)).toBe(true);
    },
  );

  it.each([
    "",
    "  ",
    "R",
    "rea",
    "REACT",
    "REACT:",
    "REACT: ",
    "REACT: <NEW_MESSAGE_BREAK>",
  ])("holds %j, which one more frame can still make a directive", (text) => {
    expect(couldBecomeReactDirective(text)).toBe(true);
  });

  it.each([
    "Really interesting",
    "REACTION: completed",
    "REACT: 👍\nand more",
    "REACT:\n",
    "hello REACT: 👍",
    "R\nEACT: 👍",
  ])("releases %j, which no later frame can make a directive", (text) => {
    expect(couldBecomeReactDirective(text)).toBe(false);
  });
});
