import random
from dataclasses import dataclass
from enum import Enum
from typing import Optional


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

@dataclass
class Survey(Action):
    surveying_for: ObjectType
    survey_start: int
    survey_end: int

@dataclass
class SurveyResult:
    number_found: int

@dataclass
class Target(Action):
    targeted_sector: int

@dataclass
class TargetResult:
    object_found: ObjectType

# I don't actually plan to implement research since I'm not sure how the
# information is generated or revealed.
@dataclass
class ResearchTopic(Action):
    topic: int

@dataclass
class LocatePlanetX(Action):
    sector: int
    prev_sector_object: ObjectType
    next_sector_object: ObjectType

@dataclass
class LocatePlanetXResult:
    correct: bool

# 12 sectors is standard, 18 sectors is expert
NUM_SECTORS = 12

def next_sector(i: int) -> int:
    return (i+1) % NUM_SECTORS

def prev_sector(i: int) -> int:
    return (i-1) % NUM_SECTORS

def try_generate_solar_system() -> Optional[SolarSystem]:
    system = [None for _ in range(0, NUM_SECTORS)]

    # Comets (2 total)
    # Each comet is located in one of these particular
    # sectors, as indicated on your note sheet:
    # Standard Mode: 2, 3, 5, 7, or 11
    # Expert Mode: 2, 3, 5, 7, 11, 13, or 17
    legal_comet_locations = [2,3,5,7,11]
    comet_locations = random.choices(legal_comet_locations, k=2)

    for comet_location in comet_locations:
        system[comet_location-1] = ObjectType.COMET

    # Asteroids (4 total)
    # Each asteroid is adjacent to at least one other
    # asteroid. (This means that the asteroids are
    # either in two separate pairs or in one group of
    # four.)

    # We're just looking for a run of two, twice.
    for _ in range(0, 2):
        run_starts = [start
                      for start in range(0, NUM_SECTORS)
                      if system[start] is None and system[next_sector(start)] is None]
        chosen_start = random.choice(run_starts)
        system[chosen_start] = ObjectType.ASTEROID
        system[next_sector(chosen_start)] = ObjectType.ASTEROID

    # Gas Clouds (2 total)
    # Each gas cloud is adjacent to at least one truly
    # empty sector.

    # Truly Empty Sectors (2 total)
    truly_empty_generated = 0

    for _ in range(0, 2):
        legal_gas_cloud_locations = [sector
                                     for sector in range(0, NUM_SECTORS)
                                     if system[sector] is None
                                     and (system[prev_sector(sector)] in {None, ObjectType.TRULY_EMPTY}
                                          or system[next_sector(sector)] in {None, ObjectType.TRULY_EMPTY})]

        chosen_sector = random.choice(legal_gas_cloud_locations)
        system[chosen_sector] = ObjectType.GAS_CLOUD

        # If there's already an empty sector next to the gas cloud, it's already legal
        if ObjectType.TRULY_EMPTY in {system[prev_sector(chosen_sector)], system[next_sector(chosen_sector)]}:
            continue

        # Else we need to pick a sector and put one there
        truly_empty_location = random.choice([prev_sector(chosen_sector), next_sector(chosen_sector)],)
        system[truly_empty_location] = ObjectType.TRULY_EMPTY
        truly_empty_generated += 1

    while truly_empty_generated < 2:
        legal_empty_locations = [sector
                                 for sector in range(0, NUM_SECTORS)
                                 if system[sector] is None]
        chosen_sector = random.choice(legal_empty_locations)
        system[chosen_sector] = ObjectType.TRULY_EMPTY
        truly_empty_generated += 1

    # There are now two spaces left. If they are adjacent, there is no legal system possible.
    empty_sectors = [sector for sector in range(0, NUM_SECTORS) if system[sector] is None]
    if abs(empty_sectors[0] - empty_sectors[1]) == 1:
        print("Illegal board state")
        return None

    # Dwarf Planets (1 total)
    # No dwarf planet is adjacent to Planet X.
    dwarf_planet_sector = random.choice(empty_sectors)
    system[dwarf_planet_sector] = ObjectType.DWARF_PLANET

    # Planet X (1 total)
    # Planet X is not adjacent to a dwarf planet.
    # In surveys and targets, the sector containing
    # Planet X appears empty.
    empty_sectors = [sector for sector in range(0, NUM_SECTORS) if system[sector] is None]
    planet_x_sector = random.choice(empty_sectors)
    system[planet_x_sector] = ObjectType.PLANET_X

    return SolarSystem(system)


@dataclass
class SearchResult:
    time_cost: int
    actions: list[Action]

def find_planet_x(solar_system: SolarSystem) -> SearchResult:
    pass

GENERATION_ATTEMPTS = 10000

def main():
    for attempt in range(0, GENERATION_ATTEMPTS):
        solar_system: Optional[SolarSystem] = try_generate_solar_system()
        if solar_system is not None:
            print(f"Generated system in {attempt+1} attempts")
            break

    if solar_system is None:
        print(f"failed to generate after {GENERATION_ATTEMPTS} attempts")
        return

    #search_result: SearchResult = find_planet_x(solar_system)

    #print(f"Found Planet X in {search_result.time_cost} time")
    #print(f"Solution: {search_result.actions}")

if __name__ == "__main__":
    main()