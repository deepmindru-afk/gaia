import { describe, expect, it } from "vitest";
import { flattenArgsPreview } from "./args-preview";

describe("flattenArgsPreview (RN port of web argsPreview)", () => {
  it("flattens nested event arrays into grouped rows (calendar create)", () => {
    const { rows } = flattenArgsPreview({
      events: [
        { title: "Dentist", start: "2026-09-21T10:00:00", location: "Clinic" },
        { title: "Lunch", start: "2026-09-21T12:30:00" },
      ],
    });
    const groups = new Set(rows.map((r) => r.group));
    expect(groups.has("Event 1")).toBe(true);
    expect(groups.has("Event 2")).toBe(true);
    const titles = rows.filter((r) => r.key === "title").map((r) => r.value);
    expect(titles).toEqual(["Dentist", "Lunch"]);
  });

  it("keeps top-level scalars and formats booleans", () => {
    const { rows } = flattenArgsPreview({
      to: "a@example.com",
      count: 3,
      urgent: true,
    });
    expect(rows).toContainEqual({
      key: "to",
      value: "a@example.com",
      group: null,
    });
    expect(rows).toContainEqual({ key: "count", value: "3", group: null });
    expect(rows).toContainEqual({ key: "urgent", value: "Yes", group: null });
  });

  it("walks one level into objects instead of dropping them", () => {
    const { rows } = flattenArgsPreview({
      event: { title: "Standup", location: "Zoom" },
    });
    expect(rows.some((r) => r.value === "Standup")).toBe(true);
    expect(rows.some((r) => r.value === "Zoom")).toBe(true);
  });

  it("returns no rows for empty args and caps runaway input", () => {
    expect(flattenArgsPreview({}).rows).toEqual([]);
    const big: Record<string, unknown> = {};
    for (let i = 0; i < 50; i++) big[`field_${i}`] = `value ${i}`;
    const { rows, omitted } = flattenArgsPreview(big);
    expect(rows.length).toBeLessThanOrEqual(20);
    expect(omitted).toBeGreaterThan(0);
  });
});
