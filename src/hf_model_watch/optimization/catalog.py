"""Read-only catalog placeholder, replaceable by an indexed manifest provider."""

from typing import Protocol

from .contracts import StudyContext, StudySummary


class StudyCatalog(Protocol):
    def list_studies(self) -> list[StudySummary]: ...
    def get_study(self, reference: str) -> StudySummary: ...


class FixtureStudyCatalog:
    def __init__(self, studies: list[StudySummary] | None = None):
        self._studies = {s.study_reference: s.model_copy(deep=True) for s in studies or []}

    def list_studies(self) -> list[StudySummary]:
        return [self._studies[k].model_copy(deep=True) for k in sorted(self._studies)]

    def get_study(self, reference: str) -> StudySummary:
        if reference not in self._studies:
            raise ValueError(f"Prior study not found: {reference}")
        return self._studies[reference].model_copy(deep=True)


def incompatibilities(expected: StudyContext, actual: StudyContext) -> list[str]:
    # Unknown metadata is rejected by the typed context before reaching this check.
    left, right = expected.model_dump(), actual.model_dump()
    for data in (left, right):
        data["constraints"] = sorted(
            data["constraints"], key=lambda b: (b["metric"], b["operator"], b["value"])
        )
    return [key for key in left if left[key] != right[key]]
