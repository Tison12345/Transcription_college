"""
Structured record of one patient follow-up call.

Two parts, kept separate on purpose:

  PatientInfo - what the patient told us (clinical content).
  CallState   - how the call is going (status and counters), plus a
                PatientInfo.

Held in memory only. Nothing here is saved or sent anywhere.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional


# conversation_status values
IN_PROGRESS = "in_progress"
COMPLETED = "completed"                  # reached the normal closing
ENDED_NO_RESPONSE = "ended_no_response"  # patient could not be heard/understood
ENDED_ERROR = "ended_error"              # technical problem ended the call


@dataclass
class PatientInfo:
    general_feeling: Optional[str] = None     # patient's own words
    pain_reported: Optional[bool] = None      # None = not known yet
    pain_location: Optional[str] = None       # e.g. "ankle"; None = unknown
    pain_description: Optional[str] = None    # patient's own words


@dataclass
class CallState:
    patient: PatientInfo = field(default_factory=PatientInfo)
    conversation_status: str = IN_PROGRESS
    clarification_count: int = 0      # clarification questions asked this call
    low_confidence_turns: int = 0     # transcripts that were not trusted

    def to_dict(self):
        """Plain dict, e.g. for a future call summary."""
        return asdict(self)
