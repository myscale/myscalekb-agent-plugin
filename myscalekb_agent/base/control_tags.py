from enum import Enum


class ControlTags(str, Enum):
    """Using tags in langgraph to control business processes is flexible but very tricky."""

    BOOTSTRAP = "bootstrap"
    SUMMARY_TAG = "summary_tag"
    EXCLUDE_STREAM = "exclude_stream"
