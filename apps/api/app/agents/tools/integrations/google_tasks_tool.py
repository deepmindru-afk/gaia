"""Google Tasks custom tools using Composio custom tool infrastructure."""

from datetime import UTC, datetime

from composio import Composio
from composio.types import ExecuteRequestFn

from app.models.common_models import GatherContextInput
from app.models.integrations.composio import CustomToolAuthCredentials
from app.models.integrations.google_tasks import GoogleTaskList
from app.utils.context_utils import execute_tool


def register_google_tasks_custom_tools(composio: Composio) -> list[str]:
    """Register Google Tasks tools as Composio custom tools."""

    @composio.tools.custom_tool(toolkit="GOOGLETASKS")
    def CUSTOM_GATHER_CONTEXT(
        request: GatherContextInput,
        execute_request: ExecuteRequestFn,
        auth_credentials: dict[str, object],
    ) -> dict[str, object]:
        """Get Google Tasks context snapshot: task lists and overdue/due-today tasks.

        Zero required parameters. Returns task lists and urgent tasks.
        """
        del request, execute_request  # unused: framework-mandated custom-tool signature
        user_id = CustomToolAuthCredentials.parse(auth_credentials).user_id

        tasks = GoogleTaskList.model_validate(
            execute_tool(
                "GOOGLETASKS_LIST_ALL_TASKS",
                {"showCompleted": False, "maxResults": 20},
                user_id,
            )
        ).all
        today = datetime.now(UTC).date().strftime("%Y-%m-%d")
        overdue = [t for t in tasks if (t.due if t.due is not None else "9999") < today]
        return {
            "tasks": [t.model_dump(mode="json", exclude_unset=True) for t in tasks],
            "overdue_tasks": [t.model_dump(mode="json", exclude_unset=True) for t in overdue],
        }

    return ["GOOGLETASKS_CUSTOM_GATHER_CONTEXT"]
