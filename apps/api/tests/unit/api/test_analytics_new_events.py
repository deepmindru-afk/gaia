"""Analytics coverage for previously-missing backend events.

Every mutating endpoint that had no PostHog event now emits exactly one,
after success, attributed to the authenticated user. Each test below drives
the real route with the service seam mocked and asserts the exact event plus
the distinct_id — a call-count-only assertion would not catch attribution to
an anonymous profile, which is the silent failure mode here.
"""

import re
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from httpx import AsyncClient
import pytest

from app.services.analytics_service import AnalyticsEvents

pytestmark = pytest.mark.unit

UID = "507f1f77bcf86cd799439011"


def test_all_event_names_are_domain_action_snake_case() -> None:
    for member in AnalyticsEvents:
        assert re.fullmatch(r"[a-z0-9_]+(:[a-z0-9_]+)?", member.value), member


# ---------------------------------------------------------------------------
# Workflows
# ---------------------------------------------------------------------------

WF = "/api/v1/workflows"
_WF_SERVICE = "app.api.v1.endpoints.workflows.WorkflowService"
_WF_REPO = "app.api.v1.endpoints.workflows.workflow_repository"
_WF_CAPTURE = "app.api.v1.endpoints.workflows.capture_context_event"


def _make_workflow(**overrides):  # type: ignore[no-untyped-def]
    from app.models.workflow_models import Workflow

    base: dict = {
        "id": "wf_abc123",
        "user_id": UID,
        "title": "My Workflow",
        "description": "A test workflow",
        "prompt": "Do the thing",
        "steps": [
            {"id": "step_1", "title": "Step 1", "category": "general", "description": "x"}
        ],
        "trigger_config": {"type": "manual", "enabled": True},
        "activated": True,
        "is_public": False,
        "slug": None,
        "total_executions": 0,
        "successful_executions": 0,
        "last_executed_at": None,
        "created_at": datetime(2025, 1, 1, tzinfo=UTC),
        "updated_at": datetime(2025, 1, 1, tzinfo=UTC),
    }
    base.update(overrides)
    return Workflow(**base)


class TestWorkflowNewEvents:
    async def test_deactivate_captures(self, client: AsyncClient) -> None:
        with (
            patch(f"{_WF_SERVICE}.deactivate_workflow", new_callable=AsyncMock) as mock_svc,
            patch(_WF_CAPTURE) as mock_capture,
        ):
            mock_svc.return_value = _make_workflow()
            resp = await client.post(f"{WF}/wf_abc123/deactivate")
        assert resp.status_code == 200
        mock_capture.assert_called_once_with(AnalyticsEvents.WORKFLOW_DEACTIVATED)

    async def test_update_captures(self, client: AsyncClient) -> None:
        with (
            patch(f"{_WF_SERVICE}.update_workflow", new_callable=AsyncMock) as mock_svc,
            patch(_WF_CAPTURE) as mock_capture,
        ):
            mock_svc.return_value = _make_workflow(title="Updated")
            resp = await client.put(f"{WF}/wf_abc123", json={"title": "Updated"})
        assert resp.status_code == 200
        mock_capture.assert_called_once_with(AnalyticsEvents.WORKFLOW_UPDATED)

    async def test_delete_captures(self, client: AsyncClient) -> None:
        with (
            patch(f"{_WF_SERVICE}.delete_workflow", new_callable=AsyncMock) as mock_svc,
            patch(_WF_CAPTURE) as mock_capture,
        ):
            mock_svc.return_value = True
            resp = await client.delete(f"{WF}/wf_abc123")
        assert resp.status_code == 200
        mock_capture.assert_called_once_with(AnalyticsEvents.WORKFLOW_DELETED)

    async def test_unpublish_captures(self, client: AsyncClient) -> None:
        from app.models.workflow_models import WorkflowDocument

        doc = WorkflowDocument(**_make_workflow(is_public=True).model_dump())
        with (
            patch(f"{_WF_REPO}.get_for_user", new_callable=AsyncMock, return_value=doc),
            patch(f"{_WF_REPO}.unpublish", new_callable=AsyncMock, return_value=doc),
            patch(_WF_CAPTURE) as mock_capture,
        ):
            resp = await client.post(f"{WF}/wf_abc123/unpublish")
        assert resp.status_code == 200
        mock_capture.assert_called_once_with(AnalyticsEvents.WORKFLOW_UNPUBLISHED)

    async def test_regenerate_captures(self, client: AsyncClient) -> None:
        with (
            patch(
                f"{_WF_SERVICE}.regenerate_workflow_steps",
                new_callable=AsyncMock,
                return_value=_make_workflow(),
            ),
            patch(_WF_CAPTURE) as mock_capture,
        ):
            resp = await client.post(
                f"{WF}/wf_abc123/regenerate-steps", json={"instruction": "Better"}
            )
        assert resp.status_code == 200
        mock_capture.assert_called_once_with(
            AnalyticsEvents.WORKFLOW_STEPS_REGENERATED,
            {"force_different_tools": False, "steps_count": 1},
        )

    async def test_from_todo_captures_created(self, client: AsyncClient) -> None:
        with (
            patch(f"{_WF_SERVICE}.create_workflow", new_callable=AsyncMock) as mock_svc,
            patch(_WF_CAPTURE) as mock_capture,
        ):
            mock_svc.return_value = _make_workflow(title="Todo: Buy groceries")
            resp = await client.post(
                f"{WF}/from-todo",
                json={"todo_id": "todo_123", "todo_title": "Buy groceries"},
            )
        assert resp.status_code == 200
        mock_capture.assert_called_once_with(
            AnalyticsEvents.WORKFLOW_CREATED, {"from_todo": True}
        )


# ---------------------------------------------------------------------------
# Device revoke (explicit distinct_id — auth-excluded pattern sibling)
# ---------------------------------------------------------------------------


class TestDeviceRevoke:
    async def test_revoke_captures_with_user_id(self, client: AsyncClient) -> None:
        with (
            patch(
                "app.api.v1.endpoints.device.revoke_device",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch("app.api.v1.endpoints.device.capture_event") as mock_capture,
        ):
            resp = await client.delete("/api/v1/device/dev-1")
        assert resp.status_code == 200
        mock_capture.assert_called_once_with(UID, AnalyticsEvents.DEVICE_REVOKED)


# ---------------------------------------------------------------------------
# Reminders
# ---------------------------------------------------------------------------

REM = "/api/v1/reminders"
_REM_SCHED = "app.api.v1.endpoints.reminders.reminder_scheduler"
_REM_CAPTURE = "app.api.v1.endpoints.reminders.capture_context_event"


def _reminder_model(reminder_id: str = "rem_1", status: str = "scheduled") -> MagicMock:
    from datetime import timedelta

    now = datetime.now(UTC)
    future = now + timedelta(days=1)
    m = MagicMock()
    m.id = reminder_id
    m.user_id = UID
    m.agent = "static"
    m.repeat = None
    m.recurrence = None
    m.scheduled_at = future
    m.next_run_time = future
    m.status = status
    m.occurrence_count = 0
    m.max_occurrences = None
    m.stop_after = None
    m.payload = {"title": "Test", "body": "Test body"}
    m.created_at = now
    m.updated_at = now
    m.model_dump.return_value = {
        "id": reminder_id,
        "user_id": UID,
        "agent": "static",
        "repeat": None,
        "scheduled_at": future,
        "status": status,
        "occurrence_count": 0,
        "max_occurrences": None,
        "stop_after": None,
        "payload": {"title": "Test", "body": "Test body"},
        "created_at": now,
        "updated_at": now,
    }
    return m


class TestReminderNewEvents:
    async def test_update_captures(self, client: AsyncClient) -> None:
        with (
            patch(f"{_REM_SCHED}.update_reminder", new_callable=AsyncMock, return_value=True),
            patch(
                f"{_REM_SCHED}.get_reminder",
                new_callable=AsyncMock,
                return_value=_reminder_model(),
            ),
            patch(_REM_CAPTURE) as mock_capture,
        ):
            resp = await client.put(f"{REM}/rem_1", json={"max_occurrences": 5})
        assert resp.status_code == 200
        mock_capture.assert_called_once_with(AnalyticsEvents.REMINDER_UPDATED)

    async def test_pause_captures(self, client: AsyncClient) -> None:
        with (
            patch(f"{_REM_SCHED}.update_reminder", new_callable=AsyncMock, return_value=True),
            patch(
                f"{_REM_SCHED}.get_reminder",
                new_callable=AsyncMock,
                return_value=_reminder_model(status="paused"),
            ),
            patch(_REM_CAPTURE) as mock_capture,
        ):
            resp = await client.post(f"{REM}/rem_1/pause")
        assert resp.status_code == 200
        mock_capture.assert_called_once_with(AnalyticsEvents.REMINDER_PAUSED)

    async def test_resume_captures(self, client: AsyncClient) -> None:
        with (
            patch(
                f"{_REM_SCHED}.get_reminder",
                new_callable=AsyncMock,
                return_value=_reminder_model(status="paused"),
            ),
            patch(f"{_REM_SCHED}.update_reminder", new_callable=AsyncMock, return_value=True),
            patch(_REM_CAPTURE) as mock_capture,
        ):
            resp = await client.post(f"{REM}/rem_1/resume")
        assert resp.status_code == 200
        mock_capture.assert_called_once_with(AnalyticsEvents.REMINDER_RESUMED)


# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------

SK = "/api/v1/skills"
_SKILLS = "app.api.v1.endpoints.skills"
_SK_CAPTURE = f"{_SKILLS}.capture_context_event"


def _make_skill():  # type: ignore[no-untyped-def]
    from app.agents.skills.models import Skill

    return Skill(
        id="sk_abc123",
        user_id=UID,
        name="my-skill",
        description="A test skill",
        target="executor",
        body_content="# My Skill",
        vfs_path="/skills/my-skill",
        enabled=True,
        source="github",
        installed_at=datetime(2025, 1, 1, tzinfo=UTC),
        files=["SKILL.md"],
    )


class TestSkillNewEvents:
    async def test_update_captures(self, client: AsyncClient) -> None:
        with (
            patch(f"{_SKILLS}.update_skill_inline", new_callable=AsyncMock) as mock_upd,
            patch(_SK_CAPTURE) as mock_capture,
        ):
            mock_upd.return_value = _make_skill()
            resp = await client.put(f"{SK}/sk_abc123", json={"description": "new"})
        assert resp.status_code == 200
        mock_capture.assert_called_once_with(AnalyticsEvents.SKILL_UPDATED)

    async def test_enable_captures(self, client: AsyncClient) -> None:
        with (
            patch(f"{_SKILLS}.enable_skill", new_callable=AsyncMock, return_value=True),
            patch(_SK_CAPTURE) as mock_capture,
        ):
            resp = await client.patch(f"{SK}/sk_abc123/enable")
        assert resp.status_code == 200
        mock_capture.assert_called_once_with(AnalyticsEvents.SKILL_ENABLED)

    async def test_disable_captures(self, client: AsyncClient) -> None:
        with (
            patch(f"{_SKILLS}.disable_skill", new_callable=AsyncMock, return_value=True),
            patch(_SK_CAPTURE) as mock_capture,
        ):
            resp = await client.patch(f"{SK}/sk_abc123/disable")
        assert resp.status_code == 200
        mock_capture.assert_called_once_with(AnalyticsEvents.SKILL_DISABLED)


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

MEM = "/api/v1/memory"
_MEM = "app.api.v1.endpoints.memory"
_MEM_CAPTURE = f"{_MEM}.capture_context_event"
_MEM_ID = "507f1f77-bcf8-6cd7-9943-9011aaaaaaaa"


class TestMemoryNewEvents:
    async def test_create_captures(self, client: AsyncClient) -> None:
        with (
            patch(f"{_MEM}.memory_engine.retain_single", new_callable=AsyncMock) as mock_ret,
            patch(_MEM_CAPTURE) as mock_capture,
        ):
            mock_ret.return_value = SimpleNamespace(
                entry=SimpleNamespace(id="mem-1", category_path="general")
            )
            resp = await client.post(MEM, json={"content": "The sky is blue"})
        assert resp.status_code == 200
        mock_capture.assert_called_once_with(AnalyticsEvents.MEMORY_CREATED)

    async def test_update_captures(self, client: AsyncClient) -> None:
        from app.models.memory_models import MemoryEntry

        with (
            patch(f"{_MEM}.memory_engine.update_memory", new_callable=AsyncMock) as mock_upd,
            patch(_MEM_CAPTURE) as mock_capture,
        ):
            mock_upd.return_value = MemoryEntry(id="mem-1", content="new")
            resp = await client.patch(f"{MEM}/{_MEM_ID}", json={"content": "new"})
        assert resp.status_code == 200
        mock_capture.assert_called_once_with(AnalyticsEvents.MEMORY_UPDATED)

    async def test_document_update_captures(self, client: AsyncClient) -> None:
        from app.models.memory_models import MemoryDocType, MemoryDocument

        with (
            patch(f"{_MEM}.memory_engine.update_document", new_callable=AsyncMock) as mock_upd,
            patch(_MEM_CAPTURE) as mock_capture,
        ):
            mock_upd.return_value = MemoryDocument(
                doc_type=MemoryDocType.USER_MD,
                content="# me",
                version=2,
                updated_at=datetime(2025, 1, 1, tzinfo=UTC),
            )
            resp = await client.put(f"{MEM}/documents/user_md", json={"content": "# me"})
        assert resp.status_code == 200
        mock_capture.assert_called_once_with(AnalyticsEvents.MEMORY_DOCUMENT_UPDATED)


# ---------------------------------------------------------------------------
# Files + calendar prefs + image
# ---------------------------------------------------------------------------


class TestFileNewEvents:
    def _doc(self, **overrides):  # type: ignore[no-untyped-def]
        from app.models.files_models import FileDocument

        data: dict = {
            "id": "0" * 24,
            "file_id": "file-001",
            "user_id": UID,
            "filename": "doc.pdf",
            "type": "application/pdf",
            "size": 10,
            "url": "https://cdn.example.com/doc.pdf",
            "public_id": "pub-id",
            "description": "desc",
            "created_at": datetime(2025, 1, 1, tzinfo=UTC),
            "updated_at": datetime(2025, 1, 1, tzinfo=UTC),
        }
        data.update(overrides)
        return FileDocument.model_validate(data)

    async def test_update_captures(self, client: AsyncClient) -> None:
        from app.services.analytics_service import AnalyticsEvents as E

        with (
            patch("app.api.v1.endpoints.file.FileService.update", new_callable=AsyncMock) as m,
            patch("app.api.v1.endpoints.file.capture_context_event") as mock_capture,
        ):
            m.return_value = self._doc(description="Updated")
            resp = await client.put("/api/v1/file-001", json={"description": "Updated"})
        assert resp.status_code == 200
        mock_capture.assert_called_once_with(E.FILE_UPDATED)

    async def test_delete_captures(self, client: AsyncClient) -> None:
        from app.models.files_models import FileDocument as _FD  # noqa: F401
        from app.schemas.file import FileDeletedResponse
        from app.services.analytics_service import AnalyticsEvents as E

        with (
            patch("app.api.v1.endpoints.file.FileService.delete", new_callable=AsyncMock) as m,
            patch("app.api.v1.endpoints.file.capture_context_event") as mock_capture,
        ):
            m.return_value = FileDeletedResponse(
                message="File deleted successfully", file_id="file-001", filename="doc.pdf"
            )
            resp = await client.delete("/api/v1/file-001")
        assert resp.status_code == 200
        mock_capture.assert_called_once_with(E.FILE_DELETED)


class TestCalendarPrefs:
    async def test_prefs_update_captures(self, client: AsyncClient) -> None:
        from app.models.calendar_models import CalendarPreferencesUpdateResponse

        with (
            patch(
                "app.api.v1.dependencies.google_scope_dependencies.check_integration_status",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch("app.api.v1.endpoints.calendar.calendar_service", new_callable=AsyncMock) as svc,
            patch("app.api.v1.endpoints.calendar.capture_context_event") as mock_capture,
        ):
            svc.update_user_calendar_preferences.return_value = (
                CalendarPreferencesUpdateResponse(message="Preferences updated")
            )
            resp = await client.put(
                "/api/v1/calendar/preferences", json={"selected_calendars": ["primary"]}
            )
        assert resp.status_code == 200
        mock_capture.assert_called_once_with(AnalyticsEvents.CALENDAR_PREFERENCES_UPDATED)


class TestImageNewEvents:
    async def test_generate_captures(self, client: AsyncClient) -> None:
        from app.models.chat_models import ImageData

        with (
            patch("app.api.v1.endpoints.image.api_generate_image", new_callable=AsyncMock) as m,
            patch("app.api.v1.endpoints.image.capture_context_event") as mock_capture,
        ):
            m.return_value = ImageData(url="https://x/y.jpg", prompt="a cat")
            resp = await client.post("/api/v1/image/generate", json={"message": "a cat"})
        assert resp.status_code == 200
        mock_capture.assert_called_once_with(AnalyticsEvents.IMAGE_GENERATED)

    async def test_describe_captures(self, client: AsyncClient) -> None:
        from app.models.image_models import ImageToTextResponse

        with (
            patch(
                "app.api.v1.endpoints.image.image_to_text_endpoint", new_callable=AsyncMock
            ) as m,
            patch("app.api.v1.endpoints.image.capture_context_event") as mock_capture,
        ):
            m.return_value = ImageToTextResponse(response="A cat")
            resp = await client.post(
                "/api/v1/image/text",
                data={"message": "what is this"},
                files={"file": ("p.png", b"bytes", "image/png")},
            )
        assert resp.status_code == 200
        mock_capture.assert_called_once_with(AnalyticsEvents.IMAGE_DESCRIBED)


# ---------------------------------------------------------------------------
# Mail triage (representative sample; remaining label/draft variants share code)
# ---------------------------------------------------------------------------

MAIL = "app.api.v1.endpoints.mail"


@pytest.fixture
def _gmail_bypass():  # type: ignore[no-untyped-def]
    with patch(
        "app.api.v1.dependencies.google_scope_dependencies.check_integration_status",
        new_callable=AsyncMock,
        return_value=True,
    ):
        yield


@pytest.mark.usefixtures("_gmail_bypass")
class TestMailNewEvents:
    async def test_mark_read_captures_count(self, client: AsyncClient) -> None:
        from app.models.mail_models import GmailMessageResource

        with (
            patch(f"{MAIL}.mark_messages_as_read", new_callable=AsyncMock) as m,
            patch(f"{MAIL}.capture_context_event") as mock_capture,
        ):
            m.return_value = [GmailMessageResource(id="m1"), GmailMessageResource(id="m2")]
            resp = await client.post("/api/v1/gmail/mark-as-read", json={"message_ids": ["m1", "m2"]})
        assert resp.status_code == 200
        mock_capture.assert_called_once_with(
            AnalyticsEvents.EMAIL_MARKED_READ, {"message_count": 2}
        )

    async def test_star_captures(self, client: AsyncClient) -> None:
        from app.models.mail_models import GmailMessageResource

        with (
            patch(f"{MAIL}.star_messages", new_callable=AsyncMock) as m,
            patch(f"{MAIL}.capture_context_event") as mock_capture,
        ):
            m.return_value = [GmailMessageResource(id="m1")]
            resp = await client.post("/api/v1/gmail/star", json={"message_ids": ["m1"]})
        assert resp.status_code == 200
        mock_capture.assert_called_once_with(AnalyticsEvents.EMAIL_STARRED, {"message_count": 1})

    async def test_trash_captures(self, client: AsyncClient) -> None:
        with (
            patch(f"{MAIL}.trash_messages", new_callable=AsyncMock) as m,
            patch(f"{MAIL}.capture_context_event") as mock_capture,
        ):
            m.return_value = [{"id": "m1"}]
            resp = await client.post("/api/v1/gmail/trash", json={"message_ids": ["m1"]})
        assert resp.status_code == 200
        mock_capture.assert_called_once_with(AnalyticsEvents.EMAIL_TRASHED, {"message_count": 1})

    async def test_create_label_captures(self, client: AsyncClient) -> None:
        from app.models.mail_models import GmailToolResult

        with (
            patch(f"{MAIL}.create_label", new_callable=AsyncMock) as m,
            patch(f"{MAIL}.capture_context_event") as mock_capture,
        ):
            m.return_value = GmailToolResult.model_validate({"id": "L1", "name": "Imp"})
            resp = await client.post("/api/v1/gmail/labels", json={"name": "Imp"})
        assert resp.status_code == 200
        mock_capture.assert_called_once_with(AnalyticsEvents.EMAIL_LABEL_CREATED)

    async def test_apply_label_captures(self, client: AsyncClient) -> None:
        from app.models.mail_models import GmailMessageResource

        with (
            patch(f"{MAIL}.apply_labels", new_callable=AsyncMock) as m,
            patch(f"{MAIL}.capture_context_event") as mock_capture,
        ):
            m.return_value = [GmailMessageResource(id="m1")]
            resp = await client.post(
                "/api/v1/gmail/messages/apply-label",
                json={"message_ids": ["m1"], "label_ids": ["L1"]},
            )
        assert resp.status_code == 200
        mock_capture.assert_called_once_with(
            AnalyticsEvents.EMAIL_LABEL_APPLIED, {"message_count": 1}
        )

    async def test_create_draft_captures(self, client: AsyncClient) -> None:
        from app.models.mail_models import GmailToolResult

        with (
            patch(f"{MAIL}.create_draft", new_callable=AsyncMock) as m,
            patch(f"{MAIL}.capture_context_event") as mock_capture,
        ):
            m.return_value = GmailToolResult.model_validate({"id": "d1", "message": {"id": "m1"}})
            resp = await client.post(
                "/api/v1/gmail/drafts",
                json={"to": ["a@t.com"], "subject": "S", "body": "B"},
            )
        assert resp.status_code == 200
        mock_capture.assert_called_once_with(AnalyticsEvents.EMAIL_DRAFT_CREATED)

    async def test_update_draft_captures(self, client: AsyncClient) -> None:
        from app.models.mail_models import GmailToolResult

        with (
            patch(f"{MAIL}.update_draft", new_callable=AsyncMock) as m,
            patch(f"{MAIL}.capture_context_event") as mock_capture,
        ):
            m.return_value = GmailToolResult.model_validate({"id": "d1", "message": {"id": "m1"}})
            resp = await client.put(
                "/api/v1/gmail/drafts/d1",
                json={"to": ["a@t.com"], "subject": "S", "body": "B"},
            )
        assert resp.status_code == 200
        mock_capture.assert_called_once_with(AnalyticsEvents.EMAIL_DRAFT_UPDATED)

    async def test_delete_draft_captures(self, client: AsyncClient) -> None:
        with (
            patch(f"{MAIL}.delete_draft", new_callable=AsyncMock, return_value=True),
            patch(f"{MAIL}.capture_context_event") as mock_capture,
        ):
            resp = await client.delete("/api/v1/gmail/drafts/d1")
        assert resp.status_code == 200
        mock_capture.assert_called_once_with(AnalyticsEvents.EMAIL_DRAFT_DELETED)


# ---------------------------------------------------------------------------
# Notifications
# ---------------------------------------------------------------------------

NOTIF = "app.api.v1.endpoints.notification"


def _notif_record():  # type: ignore[no-untyped-def]
    from datetime import UTC as _UTC
    from datetime import datetime as _dt

    from app.models.notification.notification_models import (
        NotificationContent,
        NotificationRecord,
        NotificationRequest,
        NotificationSourceEnum,
        NotificationStatus,
        NotificationType,
    )

    return NotificationRecord(
        id="n1",
        user_id=UID,
        status=NotificationStatus.READ,
        created_at=_dt(2026, 1, 1, tzinfo=_UTC),
        original_request=NotificationRequest(
            user_id=UID,
            source=NotificationSourceEnum.AI_AGENT,
            type=NotificationType.INFO,
            content=NotificationContent(title="Hi", body="Body"),
        ),
    )


class TestNotificationNewEvents:
    async def test_read_captures(self, client: AsyncClient) -> None:
        with (
            patch(f"{NOTIF}.notification_service.mark_as_read", new_callable=AsyncMock) as m,
            patch(f"{NOTIF}.capture_context_event") as mock_capture,
        ):
            m.return_value = _notif_record()
            resp = await client.post("/api/v1/notifications/n1/read")
        assert resp.status_code == 200
        mock_capture.assert_called_once_with(AnalyticsEvents.NOTIFICATION_READ, {"count": 1})

    async def test_bulk_captures(self, client: AsyncClient) -> None:
        with (
            patch(f"{NOTIF}.notification_service.bulk_actions", new_callable=AsyncMock) as m,
            patch(f"{NOTIF}.capture_context_event") as mock_capture,
        ):
            m.return_value = {"n1": True, "n2": True}
            resp = await client.post(
                "/api/v1/notifications/bulk-actions",
                json={"notification_ids": ["n1", "n2"], "action": "mark_read"},
            )
        assert resp.status_code == 200
        mock_capture.assert_called_once_with(
            AnalyticsEvents.NOTIFICATION_BULK_ACTION,
            {"action": "mark_read", "successful": 2, "total": 2},
        )

    async def test_execute_captures(self, client: AsyncClient) -> None:
        result = MagicMock()
        result.success = True
        result.message = "Done"
        result.data = {}
        with (
            patch(
                f"{NOTIF}.notification_service.execute_action",
                new_callable=AsyncMock,
                return_value=result,
            ),
            patch(f"{NOTIF}.capture_context_event") as mock_capture,
        ):
            resp = await client.post("/api/v1/notifications/n1/actions/a1/execute")
        assert resp.status_code == 200
        mock_capture.assert_called_once_with(AnalyticsEvents.NOTIFICATION_ACTION_EXECUTED)

    async def test_unsubscribe_captures_with_user_id(self, client: AsyncClient) -> None:
        with (
            patch(f"{NOTIF}.verify_unsubscribe_token", return_value=UID),
            patch(
                f"{NOTIF}.user_repository.set_channel_preferences",
                new_callable=AsyncMock,
            ),
            patch(f"{NOTIF}.capture_event") as mock_capture,
        ):
            resp = await client.post("/api/v1/notifications/unsubscribe?token=tok")
        assert resp.status_code == 200
        mock_capture.assert_called_once_with(UID, AnalyticsEvents.NOTIFICATION_UNSUBSCRIBED)


# ---------------------------------------------------------------------------
# Approvals + platform connect-init + MCP test
# ---------------------------------------------------------------------------


class TestApprovalNewEvents:
    async def test_decision_captures(self, client: AsyncClient) -> None:
        with (
            patch(
                "app.api.v1.endpoints.approvals.resolve_approval", new_callable=AsyncMock
            ),
            patch("app.api.v1.endpoints.approvals.capture_context_event") as mock_capture,
        ):
            resp = await client.post("/api/v1/approvals/a1/decision", json={"decision": "approve"})
        assert resp.status_code == 200
        mock_capture.assert_called_once_with(
            AnalyticsEvents.APPROVAL_DECIDED, {"decision": "approve"}
        )

    async def test_batch_captures(self, client: AsyncClient) -> None:
        from app.schemas.hil_schemas import BatchDecisionOutcome

        with (
            patch(
                "app.api.v1.endpoints.approvals.resolve_approvals_batch",
                new_callable=AsyncMock,
                return_value=[BatchDecisionOutcome(approval_id="a1", resolved=True)],
            ),
            patch("app.api.v1.endpoints.approvals.capture_context_event") as mock_capture,
        ):
            resp = await client.post(
                "/api/v1/approvals/batch-decision",
                json={"decisions": [{"approval_id": "a1", "decision": "approve"}]},
            )
        assert resp.status_code == 200
        mock_capture.assert_called_once_with(
            AnalyticsEvents.APPROVAL_DECIDED,
            {"batch": True, "decisions": 1, "resolved": 1},
        )


class TestPlatformConnectInit:
    async def test_init_captures(self, client: AsyncClient) -> None:
        from app.models.platform_models import InitiatePlatformConnectResponse

        with (
            patch(
                "app.api.v1.endpoints.platform_links.start_platform_connect",
                new_callable=AsyncMock,
            ) as m,
            patch("app.api.v1.endpoints.platform_links.capture_context_event") as mock_capture,
        ):
            m.return_value = InitiatePlatformConnectResponse(
                auth_type="manual", action_link="https://t.me/x"
            )
            resp = await client.post("/api/v1/platform-links/telegram/connect", json={})
        assert resp.status_code == 200
        mock_capture.assert_called_once_with(
            AnalyticsEvents.INTEGRATION_CONNECT_INITIATED,
            {"integration_id": "telegram", "auth_type": "manual"},
        )


class TestMcpConnectionTested:
    async def test_connected_captures(self, client: AsyncClient) -> None:
        probe_client = AsyncMock()
        probe_client.probe_connection.return_value = {"requires_auth": False}
        probe_client.connect.return_value = [{"name": "t"}]
        with (
            patch(
                "app.api.v1.endpoints.mcp.get_mcp_client",
                new_callable=AsyncMock,
                return_value=probe_client,
            ),
            patch(
                "app.api.v1.endpoints.mcp.IntegrationResolver.resolve",
                new_callable=AsyncMock,
                return_value=SimpleNamespace(
                    mcp_config=SimpleNamespace(server_url="https://mcp.example.com")
                ),
            ),
            patch(
                "app.api.v1.endpoints.mcp.invalidate_user_integration_caches",
                new_callable=AsyncMock,
            ),
            patch("app.api.v1.endpoints.mcp.capture_context_event") as mock_capture,
        ):
            resp = await client.post("/api/v1/mcp/test/gh")
        assert resp.status_code == 200
        mock_capture.assert_called_once_with(
            AnalyticsEvents.MCP_CONNECTION_TESTED, {"status": "connected", "tools_count": 1}
        )


# ---------------------------------------------------------------------------
# Todos: bulk-move, projects, subtasks
# ---------------------------------------------------------------------------

TODOS = "app.api.v1.endpoints.todos"
_TODOS_CAPTURE = f"{TODOS}.capture_context_event"


class TestTodoNewEvents:
    async def test_bulk_move_captures(self, client: AsyncClient) -> None:
        from app.models.todo_models import BulkOperationResponse

        with (
            patch(f"{TODOS}.TodoService.bulk_move_todos", new_callable=AsyncMock) as m,
            patch(_TODOS_CAPTURE) as mock_capture,
        ):
            m.return_value = BulkOperationResponse(success=["t1"], total=1, message="ok")
            resp = await client.post(
                "/api/v1/todos/bulk/move", json={"todo_ids": ["t1"], "project_id": "p1"}
            )
        assert resp.status_code == 200
        mock_capture.assert_called_once_with(AnalyticsEvents.TODO_UPDATED, {"bulk_count": 1})

    async def test_project_create_captures(self, client: AsyncClient) -> None:
        from app.models.todo_models import ProjectResponse

        with (
            patch(f"{TODOS}.ProjectService.create_project", new_callable=AsyncMock) as m,
            patch(_TODOS_CAPTURE) as mock_capture,
        ):
            m.return_value = ProjectResponse(
                id="p1",
                user_id=UID,
                name="Work",
                created_at=datetime(2025, 1, 1, tzinfo=UTC),
                updated_at=datetime(2025, 1, 1, tzinfo=UTC),
            )
            resp = await client.post("/api/v1/projects", json={"name": "Work"})
        assert resp.status_code == 201
        mock_capture.assert_called_once_with(AnalyticsEvents.PROJECT_CREATED)

    async def test_project_update_captures(self, client: AsyncClient) -> None:
        from app.models.todo_models import ProjectResponse

        with (
            patch(f"{TODOS}.ProjectService.update_project", new_callable=AsyncMock) as m,
            patch(_TODOS_CAPTURE) as mock_capture,
        ):
            m.return_value = ProjectResponse(
                id="p1",
                user_id=UID,
                name="Renamed",
                created_at=datetime(2025, 1, 1, tzinfo=UTC),
                updated_at=datetime(2025, 1, 1, tzinfo=UTC),
            )
            resp = await client.put("/api/v1/projects/p1", json={"name": "Renamed"})
        assert resp.status_code == 200
        mock_capture.assert_called_once_with(AnalyticsEvents.PROJECT_UPDATED)

    async def test_project_delete_captures(self, client: AsyncClient) -> None:
        with (
            patch(f"{TODOS}.ProjectService.delete_project", new_callable=AsyncMock),
            patch(_TODOS_CAPTURE) as mock_capture,
        ):
            resp = await client.delete("/api/v1/projects/p1")
        assert resp.status_code == 204
        mock_capture.assert_called_once_with(AnalyticsEvents.PROJECT_DELETED)

    def _doc(self, **overrides):  # type: ignore[no-untyped-def]
        from app.models.todo_models import SubTask, TodoDocument

        base: dict = {
            "id": "todo-1",
            "user_id": UID,
            "title": "Test",
            "created_at": datetime(2025, 1, 1, tzinfo=UTC),
            "updated_at": datetime(2025, 1, 1, tzinfo=UTC),
            "subtasks": [SubTask(id="sub-1", title="Milk", completed=False)],
        }
        base.update(overrides)
        return TodoDocument(**base)

    async def test_subtask_add_captures(self, client: AsyncClient) -> None:
        with (
            patch(f"{TODOS}.todo_repository.add_subtask", new_callable=AsyncMock) as m,
            patch(_TODOS_CAPTURE) as mock_capture,
        ):
            m.return_value = self._doc()
            resp = await client.post("/api/v1/todos/todo-1/subtasks", json={"title": "Milk"})
        assert resp.status_code == 201
        mock_capture.assert_called_once_with(AnalyticsEvents.TODO_UPDATED, {"is_subtask": True})


# ---------------------------------------------------------------------------
# Onboarding submits
# ---------------------------------------------------------------------------

ONB = "app.api.v1.endpoints.onboarding"
_ONB_CAPTURE = f"{ONB}.capture_context_event"


class TestOnboardingNewEvents:
    async def test_integrations_submit_captures(self, client: AsyncClient) -> None:
        from app.models.user_models import OnboardingIntegrationsStatus

        with (
            patch(f"{ONB}.submit_onboarding_integrations", new_callable=AsyncMock) as m,
            patch(_ONB_CAPTURE) as mock_capture,
        ):
            m.return_value = OnboardingIntegrationsStatus.QUEUED
            resp = await client.post(
                "/api/v1/onboarding/integrations", json={"selected_integrations": ["gmail"]}
            )
        assert resp.status_code == 200
        mock_capture.assert_called_once_with(
            AnalyticsEvents.ONBOARDING_INTEGRATIONS_SUBMITTED, {"integration_count": 1}
        )

    async def test_reset_captures(self, client: AsyncClient) -> None:
        with (
            patch(f"{ONB}.reset_onboarding", new_callable=AsyncMock) as m,
            patch(_ONB_CAPTURE) as mock_capture,
        ):
            m.return_value = SimpleNamespace(
                model_dump=lambda: {
                    "workflows_deleted": 0,
                    "todos_deleted": 0,
                    "conversation_deleted": 0,
                    "demo_conversations_deleted": 0,
                    "integrations_disconnected": 0,
                    "memories_cleared": 0,
                }
            )
            resp = await client.post("/api/v1/onboarding/reset")
        assert resp.status_code == 200
        mock_capture.assert_called_once_with(AnalyticsEvents.ONBOARDING_RESET)

    async def test_writing_style_captures_length_only(self, client: AsyncClient) -> None:
        with (
            patch(f"{ONB}.save_user_edited_summary", new_callable=AsyncMock),
            patch(_ONB_CAPTURE) as mock_capture,
        ):
            resp = await client.post(
                "/api/v1/onboarding/writing-style", json={"edited_summary": "Be brief"}
            )
        assert resp.status_code == 200
        mock_capture.assert_called_once_with(
            AnalyticsEvents.ONBOARDING_WRITING_STYLE_SAVED, {"summary_length": 8}
        )
        assert "Be brief" not in str(mock_capture.call_args)

    async def test_regenerate_example_captures(self, client: AsyncClient) -> None:
        with (
            patch(
                f"{ONB}.regenerate_example_for_style",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(_ONB_CAPTURE) as mock_capture,
        ):
            resp = await client.post(
                "/api/v1/onboarding/writing-style/regenerate-example",
                json={"edited_summary": "Be brief"},
            )
        assert resp.status_code == 200
        mock_capture.assert_called_once_with(
            AnalyticsEvents.ONBOARDING_WRITING_STYLE_EXAMPLE_REGENERATED
        )

    async def test_social_profiles_captures_platforms_only(self, client: AsyncClient) -> None:
        with (
            patch(f"{ONB}.save_confirmed_profiles", new_callable=AsyncMock),
            patch(_ONB_CAPTURE) as mock_capture,
        ):
            resp = await client.post(
                "/api/v1/onboarding/social-profiles",
                json={"profiles": [{"platform": "github", "url": "https://github.com/me"}]},
            )
        assert resp.status_code == 200
        mock_capture.assert_called_once_with(
            AnalyticsEvents.ONBOARDING_SOCIAL_PROFILES_CONFIRMED,
            {"profile_count": 1, "platforms": ["github"]},
        )
        assert "github.com/me" not in str(mock_capture.call_args)


# ---------------------------------------------------------------------------
# Custom + user integrations, skills search stays client-side (local filter)
# ---------------------------------------------------------------------------

CUSTOM = "app.api.v1.endpoints.integrations.custom"
_CUSTOM_CAPTURE = f"{CUSTOM}.capture_context_event"
USERINT = "app.api.v1.endpoints.integrations.user"
_USERINT_CAPTURE = f"{USERINT}.capture_context_event"


def _custom_integration():  # type: ignore[no-untyped-def]
    from app.models.integration_models import Integration

    return Integration.model_validate(
        {
            "integration_id": "i1",
            "name": "My Tool",
            "description": "A custom integration",
            "category": "custom",
            "managed_by": "mcp",
            "source": "custom",
            "is_public": True,
            "mcp_config": {
                "server_url": "https://mcp.example.com",
                "requires_auth": False,
                "auth_type": "none",
            },
        }
    )


class TestCustomIntegrationNewEvents:
    async def test_update_captures(self, client: AsyncClient) -> None:
        with (
            patch(f"{CUSTOM}.update_custom_integration", new_callable=AsyncMock) as m,
            patch(_CUSTOM_CAPTURE) as mock_capture,
        ):
            m.return_value = _custom_integration()
            resp = await client.patch(f"/api/v1/integrations/custom/i1", json={"name": "R"})
        assert resp.status_code == 200
        mock_capture.assert_called_once_with(AnalyticsEvents.INTEGRATION_CUSTOM_UPDATED)

    async def test_delete_captures(self, client: AsyncClient) -> None:
        with (
            patch(f"{CUSTOM}.delete_custom_integration", new_callable=AsyncMock, return_value=True),
            patch(_CUSTOM_CAPTURE) as mock_capture,
        ):
            resp = await client.delete("/api/v1/integrations/custom/i1")
        assert resp.status_code == 200
        mock_capture.assert_called_once_with(AnalyticsEvents.INTEGRATION_CUSTOM_DELETED)

    async def test_publish_captures(self, client: AsyncClient) -> None:
        with (
            patch(f"{CUSTOM}.publish_custom_integration", new_callable=AsyncMock) as m,
            patch(_CUSTOM_CAPTURE) as mock_capture,
        ):
            m.return_value = {"integration_id": "i1", "public_url": "https://x/y"}
            resp = await client.post("/api/v1/integrations/custom/i1/publish")
        assert resp.status_code == 200
        mock_capture.assert_called_once_with(AnalyticsEvents.INTEGRATION_CUSTOM_PUBLISHED)

    async def test_unpublish_captures(self, client: AsyncClient) -> None:
        with (
            patch(f"{CUSTOM}.unpublish_custom_integration", new_callable=AsyncMock) as m,
            patch(_CUSTOM_CAPTURE) as mock_capture,
        ):
            m.return_value = {"integration_id": "i1"}
            resp = await client.post("/api/v1/integrations/custom/i1/unpublish")
        assert resp.status_code == 200
        mock_capture.assert_called_once_with(AnalyticsEvents.INTEGRATION_CUSTOM_UNPUBLISHED)


class TestInstructionsUpdate:
    async def test_instructions_update_captures(self, client: AsyncClient) -> None:
        with (
            patch(
                f"{USERINT}.check_user_has_integration",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(f"{USERINT}.upsert_instructions", new_callable=AsyncMock) as m,
            patch(_USERINT_CAPTURE) as mock_capture,
        ):
            m.return_value = SimpleNamespace(
                integration_id="gmail",
                content="Be brief",
                updated_by="user",
                updated_at=datetime(2025, 1, 1, tzinfo=UTC),
            )
            resp = await client.put(
                "/api/v1/integrations/users/me/integrations/gmail/instructions",
                json={"content": "Be brief"},
            )
        assert resp.status_code == 200
        mock_capture.assert_called_once_with(AnalyticsEvents.INTEGRATION_INSTRUCTIONS_UPDATED)
