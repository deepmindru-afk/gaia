// @vitest-environment jsdom

import { describe, expect, it } from "vitest";
import { flattenArgsPreview } from "@/features/chat/utils/argsPreview";

const calendarArgs = {
  confirm_immediately: true,
  events: [
    {
      calendar_id: "primary",
      summary: "Team standup",
      description: "Daily sync",
      start_datetime: "2026-09-21T10:00:00",
      duration_hours: 0,
      duration_minutes: 30,
      location: "Meet",
      attendees: ["a@x.com", "b@x.com"],
      is_all_day: false,
      create_meeting_room: false,
    },
    {
      calendar_id: "primary",
      summary: "1:1 with Priya",
      description: null,
      start_datetime: "2026-09-21T14:00:00",
      duration_hours: 1,
      duration_minutes: 0,
      location: "",
      attendees: [],
      is_all_day: false,
    },
  ],
};

describe("flattenArgsPreview", () => {
  it("shows nested event content instead of one top-level scalar", () => {
    const { rows, omitted } = flattenArgsPreview(calendarArgs);
    const keys = rows.map((row) => `${row.group ?? ""}:${row.key}`);
    expect(keys).toContain(":confirm immediately");
    expect(keys).toContain("Event 1:summary");
    expect(keys).toContain("Event 1:start datetime");
    expect(keys).toContain("Event 1:attendees");
    expect(keys).toContain("Event 2:summary");
    // Null description, empty location/attendees of event 2 are skipped.
    expect(keys).not.toContain("Event 2:description");
    expect(keys).not.toContain("Event 2:location");
    expect(omitted).toBe(0);
  });

  it("humanizes datetimes, booleans and lists", () => {
    const { rows, omitted } = flattenArgsPreview(calendarArgs);
    expect(omitted).toBe(0);
    const byKey = new Map(
      rows.map((row) => [`${row.group ?? ""}:${row.key}`, row.value]),
    );
    expect(byKey.get("Event 1:start datetime")).toMatch(/Sep 21, 10:00 AM/);
    expect(byKey.get(":confirm immediately")).toBe("Yes");
    expect(byKey.get("Event 1:attendees")).toBe("a@x.com, b@x.com");
  });

  it("never crashes on hostile shapes", () => {
    expect(flattenArgsPreview({}).rows).toEqual([]);
    expect(flattenArgsPreview({ a: null, b: undefined, c: "", d: [] }).rows).toEqual([]);
    expect(
      flattenArgsPreview({ deep: { deeper: { deepest: { x: 1 } } } }).omitted,
    ).toBeGreaterThanOrEqual(1);
    const many = flattenArgsPreview({
      events: Array.from({ length: 10 }, (_, i) => ({ summary: `E${i}` })),
    });
    expect(many.rows.length).toBeLessThanOrEqual(20);
    expect(many.omitted).toBeGreaterThan(0);
  });
});
