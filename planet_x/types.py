from dataclasses import dataclass
from enum import Enum


class ObjectType(Enum):
    COMET = 1
    ASTEROID = 2
    GAS_CLOUD = 3
    TRULY_EMPTY = 4
    DWARF_PLANET = 5
    PLANET_X = 6


@dataclass
class SolarSystem:
    sector_objects: list[ObjectType]


class Action:
    pass


@dataclass(frozen=True, eq=True)
class Survey(Action):
    surveying_for: ObjectType
    survey_start: int
    survey_size: int


@dataclass
class SurveyResult:
    number_found: int


# I don't actually plan to implement research since I'm not sure how the
# information is generated or revealed.

@dataclass
class LocatePlanetX(Action):
    sector: int
    prev_sector_object: ObjectType
    next_sector_object: ObjectType


@dataclass
class SearchResult:
    time_cost: int
    actions: list[Action]
