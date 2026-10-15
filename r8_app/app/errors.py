"""Errors an agent can raise that the API should show to the user as-is."""


class AgentError(Exception):
    """An expected failure inside an agent (missing dependency, LLM request failed, ...).

    The API turns it into a 502 for `/ask` and an `error` event on `/ask/stream`, so the
    message must be safe and useful to show in the web app.
    """
