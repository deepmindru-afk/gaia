/**
 * REACT-directive suppression in the shared bot streamer (`streamChat`).
 *
 * The server streams comms' `REACT: <emoji>` control line as ordinary text,
 * then follows with an `emoji_ack` frame carrying the bare emoji. Per-chunk
 * adapters (Slack `chat.update`, Telegram `editMessageText`) paint every
 * `onChunk` immediately, so forwarding the directive leaks `REACT: 😎` into
 * the chat. The streamer must hold directive-shaped text back and deliver the
 * emoji instead — on every platform, streaming or render-at-end.
 *
 * Only the axios transport is faked; the streaming/parsing code is real.
 */
import { Readable } from "node:stream";
import { describe, expect, it, vi } from "vitest";
import type { ChatStreamClient } from "../../../../../libs/shared/ts/src/bots/api/chat-stream";
import {
  isReactDirectiveOrPrefix,
  streamChat,
} from "../../../../../libs/shared/ts/src/bots/api/chat-stream";
import type { ChatRequest } from "../../../../../libs/shared/ts/src/bots/types";

const REQUEST: ChatRequest = {
  message: "hi",
  platform: "slack",
  platformUserId: "U123",
  channelId: "C123",
};

function makeDeps(sseBody: string): ChatStreamClient {
  return {
    client: {
      post: vi.fn(async () => ({ data: Readable.from([sseBody]) })),
    } as unknown as ChatStreamClient["client"],
    userHeaders: () => ({}),
    storeSessionToken: vi.fn(),
    clearSessionToken: vi.fn(),
  };
}

/** Drives one scripted SSE body through the real streamer. */
async function drive(sseBody: string) {
  const onChunk = vi.fn();
  const onDone = vi.fn();
  const onError = vi.fn();
  await streamChat(
    makeDeps(sseBody),
    REQUEST,
    onChunk,
    onDone,
    onError,
    "/api/v1/bot/chat-stream",
    vi.fn(),
    vi.fn(),
    vi.fn(),
  );
  return { onChunk, onDone, onError };
}

function frames(...payloads: object[]): string {
  return payloads.map((p) => `data: ${JSON.stringify(p)}\n\n`).join("");
}

describe("isReactDirectiveOrPrefix", () => {
  it.each(["REACT: 👍", "react: 👍", "  REACT:  ✅  "])(
    "matches the directive %j",
    (text) => {
      expect(isReactDirectiveOrPrefix(text)).toBe(true);
    },
  );

  it.each(["R", "RE", "REA", "REAC", "REACT", "REACT:", "REACT: "])(
    "holds the split prefix %j until the next frame disambiguates it",
    (text) => {
      expect(isReactDirectiveOrPrefix(text)).toBe(true);
    },
  );

  it.each([
    "Really interesting",
    "REACTION: completed",
    "REACT: 👍\nand more",
    "hello REACT: 👍",
  ])("forwards the ordinary reply %j", (text) => {
    expect(isReactDirectiveOrPrefix(text)).toBe(false);
  });
});

describe("streamChat — REACT directive suppression", () => {
  it("never forwards the directive, split across chunks, and delivers the emoji", async () => {
    const { onChunk, onDone, onError } = await drive(
      frames(
        { text: "REACT:" },
        { text: " 😎" },
        { emoji_ack: { emoji: "😎", reacts_to_message_id: "u1" } },
        { done: true, conversation_id: "c1" },
      ),
    );

    expect(onError).not.toHaveBeenCalled();
    // The only chunk any adapter ever sees is the emoji itself.
    expect(onChunk).toHaveBeenCalledTimes(1);
    expect(onChunk).toHaveBeenCalledWith("😎");
    // Render-at-end platforms (Discord/WhatsApp) deliver from fullText.
    expect(onDone).toHaveBeenCalledWith("😎", "c1");
  });

  it("suppresses a single-frame directive the same way", async () => {
    const { onChunk, onDone } = await drive(
      frames(
        { text: "REACT: 👍" },
        { emoji_ack: { emoji: "👍", reacts_to_message_id: "u1" } },
        { done: true, conversation_id: "c1" },
      ),
    );

    expect(onChunk).toHaveBeenCalledTimes(1);
    expect(onChunk).toHaveBeenCalledWith("👍");
    expect(onDone).toHaveBeenCalledWith("👍", "c1");
  });

  it("holds a lookalike prefix, then flushes it whole once disambiguated", async () => {
    const { onChunk, onDone } = await drive(
      frames(
        { text: "REAC" },
        { text: "TION: completed" },
        { done: true, conversation_id: "c1" },
      ),
    );

    // "REAC" could still become the directive, so nothing is forwarded until
    // "TION: completed" proves it is an ordinary reply — then all of it at once.
    expect(onChunk.mock.calls.flat()).toEqual(["REACTION: completed"]);
    expect(onDone.mock.calls[0][0]).toBe("REACTION: completed");
  });

  it("still forwards an ordinary reply untouched", async () => {
    const { onChunk, onDone } = await drive(
      frames(
        { text: "Here is " },
        { text: "your answer." },
        { done: true, conversation_id: "c1" },
      ),
    );

    expect(onChunk.mock.calls.flat()).toEqual(["Here is ", "your answer."]);
    expect(onDone.mock.calls[0][0]).toBe("Here is your answer.");
  });
});
