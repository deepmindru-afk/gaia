"""VFS-related constants."""

SYSTEM_USER_ID = "system"

# One path COMPONENT: a user/conversation id that is interpolated into a JuiceFS
# path must not be able to escape it. Lives here rather than beside the mount
# helpers because app.models.message_models constrains a field with it, and a
# model reaching into app.services.storage for a regex dragged the whole storage
# tree into every importer of a chat model.
SAFE_PATH_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"
