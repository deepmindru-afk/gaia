"""Registry keys shared by the analytics wiring and the modules that read it."""

#: The lazy-provider registry key of the shared PostHog client. Lives here, not
#: in app.config.posthog, so a module the embedding sidecar imports can name it
#: without loading the settings that config module validates on import.
POSTHOG_PROVIDER_KEY = "posthog"
