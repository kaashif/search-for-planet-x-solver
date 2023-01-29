import random
from z3 import z3
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

NUM_VISIBLE_SECTORS = NUM_SECTORS/2
def get_possible_actions(visible_range_start: int) -> list[Action]:
    # You can't survey for Planet X
    surveyable_objects = set(ObjectType)
    surveyable_objects.remove(ObjectType.PLANET_X)

    # We're only going to do e.g. survey 1-4, 2-5, 3-6.
    # TODO: Do the optimally sized survey instead of this.
    # Right now this seems fine since narrower surveys seem to be too expensive, while wider surveys provide too little
    # information. This opinion is subject to change.
    survey_size = 4

    return [
        Survey(survey_object, start, start+survey_size-1)
        for survey_object in surveyable_objects
        for start in range(visible_range_start, visible_range_start+3)
    ]

def pick_best_action(actions: list[Action], information: list[SurveyResult]) -> Action:
    # TODO: put a real strategy here using information theory or something cool
    # This is obviously complete shit and uses no information.
    return random.choice(actions)

def execute_action(action: Action, solar_system: SolarSystem) -> SurveyResult:
    assert isinstance(action, Survey)
    found = 0

    for sector in range(action.survey_start, action.survey_end+1):
        if solar_system.sector_objects[sector] == action.surveying_for:
            found += 1

    return SurveyResult(found)


def deduce_planet_x_location(information: list[tuple[Survey, SurveyResult]]) -> Optional[LocatePlanetX]:
    # We've found Planet X if among all solutions matching our information, all have the
    # same Planet X location.

    # let each X_i represent the object present in the solar system at sector i
    X = [z3.Int(f"X_{i}") for i in range(0, NUM_SECTORS)]

    def model_to_planet_x_location(model: z3.Model) -> LocatePlanetX:
        for i in range(0, NUM_SECTORS):
            if model[X[i]] == ObjectType.PLANET_X.value:
                return LocatePlanetX(ObjectType(model[X[i]]),
                                     ObjectType(model[X[prev_sector(i)]]),
                                     ObjectType(model[X[next_sector(i)]]))

        raise Exception("No Planet X found in model???")

    constraints = []

    # Ensure all variables have valid object type values
    for i in range(0, NUM_SECTORS):
        constraints.append(X[i] >= 1)
        constraints.append(X[i] <= 6)

    # Ensure object counts are correct
    def add_object_limit_in_range(obj_type: ObjectType, count: int, sectors: range):
        constraints.append(
            z3.Sum(z3.If(X[i] == obj_type.value, 1, 0) for i in sectors) <= count
        )

    def add_total_object_limit(obj_type: ObjectType, count: int):
        add_object_limit_in_range(obj_type, count, range(0, NUM_SECTORS))

    add_total_object_limit(ObjectType.PLANET_X, 1)
    add_total_object_limit(ObjectType.DWARF_PLANET, 1)
    add_total_object_limit(ObjectType.GAS_CLOUD, 2)
    add_total_object_limit(ObjectType.TRULY_EMPTY, 2)
    add_total_object_limit(ObjectType.COMET, 2)
    add_total_object_limit(ObjectType.ASTEROID, 4)

    # Object relationships need to be correct
    # TODO: write the object relationships as z3 constraints

    # The real meat - our survey data needs to be taken into account
    for survey, survey_result in information:
        add_object_limit_in_range(survey.surveying_for,
                                  survey_result.number_found,
                                  range(survey.survey_start, survey.survey_end+1))

    solver = z3.Solver()
    solver.add(*constraints)

    # We collect one model exhibiting each possible Planet X location
    # If there's only one, we've found it!
    models = []
    while solver.check():
        model = solver.model()
        models.append(model)
        location = model_to_planet_x_location(model)
        solver.add(X[location])

    if len(models) == 1:
        return model_to_planet_x_location(models[0])

    return None
def find_planet_x(solar_system: SolarSystem) -> SearchResult:
    # The strategy here is:
    # * Always survey 4 sectors (narrowest 3 cost search)
    # * Never target (because it seems too expensive compared to survey + perfect deduction)
    # * Never research (because I don't know how the research results are generated)
    # * Only locate Planet X when we're 100% sure
    time = 0
    actions: list[Action] = []
    information: list[SurveyResult] = []
    visible_range_start = 0

    while True:
        # TODO: This is dumb, it should be smarter
        possible_actions = get_possible_actions(visible_range_start)
        action = pick_best_action(possible_actions, information)
        actions.append(action)

        if isinstance(action, Survey):
            information.append(execute_action(action, solar_system))

            # cost of 4 width survey
            # TODO: don't hardcode time values
            time += 3
            visible_range_start = (visible_range_start+3) % NUM_SECTORS

        planet_x_location: Optional[LocatePlanetX] = deduce_planet_x_location(information)
        if planet_x_location is not None:
            actions.append(planet_x_location)
            time += 5
            break

    return SearchResult(time, actions)

GENERATION_ATTEMPTS = 10000

def main():
    for attempt in range(0, GENERATION_ATTEMPTS):
        solar_system: Optional[SolarSystem] = try_generate_solar_system()
        if solar_system is not None:
            print(f"Generated system in {attempt+1} attempts")
            break

    if solar_system is None:
        print(f"Failed to generate after {GENERATION_ATTEMPTS} attempts")
        return

    print("Board:")
    for i in range(0, NUM_SECTORS):
        print(f"{i+1}: {solar_system.sector_objects[i].name}")

    search_result: SearchResult = find_planet_x(solar_system)

    print(f"Found Planet X in {search_result.time_cost} time")
    print(f"Solution: {search_result.actions}")

if __name__ == "__main__":
    main()