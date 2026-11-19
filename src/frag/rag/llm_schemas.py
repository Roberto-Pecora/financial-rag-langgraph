"""Pydantic models for the LLM boundary — the JSON the actor and critic emit.

model_validate_json() does decode, shape-check, coercion and defaults in one
call, raising ValidationError for anything malformed.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class ActorResponse(BaseModel):
    # `answer` is required: a missing field must fail validation (and abstain upstream)
    # rather than default to "" and be accepted as an empty answer.
    answer: str
    citations: list[str] = Field(default_factory=list)

    @field_validator("answer", mode="before")
    @classmethod
    def _as_stripped_str(cls, v: object) -> str:
        return str(v).strip() if v is not None else ""

    @field_validator("citations", mode="before")
    @classmethod
    def _as_str_list(cls, v: object) -> list[str]:
        if not isinstance(v, list):
            return []
        return [str(c).strip() for c in v]


class CriticResponse(BaseModel):
    # Scores are required and bounded: a missing score must raise ValidationError (the
    # node reports "critic failed") instead of defaulting to 0.0, which was
    # indistinguishable from a genuine veto; an out-of-range score must not slip past
    # the `>= critic_min_score` gate.
    overall_score: float = Field(ge=0.0, le=1.0)
    faithfulness_score: float = Field(ge=0.0, le=1.0)
    completeness_score: float = Field(ge=0.0, le=1.0)
    citation_score: float = Field(ge=0.0, le=1.0)
    issues: list[str] = Field(default_factory=list)

    @field_validator("issues", mode="before")
    @classmethod
    def _as_str_list(cls, v: object) -> list[str]:
        if v is None:
            return []
        if not isinstance(v, list):
            return [str(v)]
        return [str(i) for i in v]
