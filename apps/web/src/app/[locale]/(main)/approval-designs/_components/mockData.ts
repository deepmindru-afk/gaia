import type { ApprovalRequestData } from "@shared/chat";

/** The hard case: nested events array — the shape that showed 1 arg before. */
export const calendarCard: ApprovalRequestData = {
  approval_id: "ap_9f2k41",
  tool_call_id: "call-7",
  gated_tool_name: "GOOGLECALENDAR_CUSTOM_CREATE_EVENT",
  integration_name: "Google Calendar",
  summary: "Create 2 events (Google Calendar)",
  args_preview: {
    events: [
      {
        calendar_id: "primary",
        summary: "Team standup",
        description: "Daily engineering sync",
        start_datetime: "2026-09-21T10:00:00",
        duration_hours: 0,
        duration_minutes: 30,
        location: "Google Meet",
        attendees: ["dhruv@x.com", "priya@x.com", "sam@x.com"],
        is_all_day: false,
        create_meeting_room: false,
      },
      {
        calendar_id: "primary",
        summary: "1:1 with Priya",
        description: "Career check-in",
        start_datetime: "2026-09-21T14:00:00",
        duration_hours: 1,
        duration_minutes: 0,
        location: "Room 3B",
        attendees: ["priya@x.com"],
        is_all_day: false,
        create_meeting_room: false,
      },
    ],
  },
  status: "pending",
  feedback: null,
  timeout_seconds: 300,
  rationale: "You asked me to schedule Tuesday's meetings.",
  age_seconds: 320,
  ledger_version: 0,
};

/** The simple case: flat scalars, for contrast. */
export const gmailCard: ApprovalRequestData = {
  approval_id: "ap_77qx20",
  tool_call_id: "call-9",
  gated_tool_name: "GMAIL_SEND_EMAIL",
  integration_name: "Gmail",
  summary: "Send email to iso@x.com (Gmail)",
  args_preview: {
    to: "iso@x.com",
    subject: "Q3 invoice attached",
    body: "Hi Iso — attaching the Q3 invoice. Let me know if anything looks off.",
  },
  status: "pending",
  feedback: null,
  timeout_seconds: 300,
  rationale: null,
  age_seconds: 9000,
  ledger_version: 2,
};
