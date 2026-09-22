"""Comms directive protocol: control keywords and the outcomes they map to.

The keywords comms emits when a background executor update is not worth a full
message. Shared by the narration prompt note (which instructs comms) and the
parser (which interprets comms' output) so the two can never drift apart.
"""

from enum import StrEnum

#: The control keywords comms may emit as its entire narration message.
SILENCE_KEYWORD = "SILENCE"
REACT_KEYWORD = "REACT"


class CommsDirectiveKind(StrEnum):
    REPLY = "reply"
    SILENCE = "silence"
    REACT = "react"
