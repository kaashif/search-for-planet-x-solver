import random

from z3 import z3
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Iterable


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
    return (i + 1) % NUM_SECTORS


def prev_sector(i: int) -> int:
    return (i - 1) % NUM_SECTORS


def object_limit_in_range_constraint(X: list[z3.Int], obj_type: ObjectType, count: int, sectors: range):
    return sum(z3.If(X[i] == obj_type.value, 1, 0) for i in sectors) <= count


def total_object_limit_constraint(X: list[z3.Int], obj_type: ObjectType, count: int):
    return object_limit_in_range_constraint(X, obj_type, count, range(0, NUM_SECTORS))


def value_in_set_constraint(x: z3.Int, value_set: Iterable[int]):
    return z3.Or([x == value for value in value_set])


def get_base_system_constraints():
    X = [z3.BitVec(f"X_{i}", 4) for i in range(0, NUM_SECTORS)]
    constraints = []

    # There's only one global constraint: the numbers of objects.
    constraints += map(lambda args: total_object_limit_constraint(X, *args), [
        (ObjectType.PLANET_X, 1),
        (ObjectType.DWARF_PLANET, 1),
        (ObjectType.GAS_CLOUD, 2),
        (ObjectType.TRULY_EMPTY, 2),
        (ObjectType.COMET, 2),
        (ObjectType.ASTEROID, 4)
    ])

    # Local constraints
    for i in range(0, NUM_SECTORS):
        # Ensure all variables have valid object type values
        constraints.append(X[i] >= 1)
        constraints.append(X[i] <= 6)

        # Comets (2 total)
        # Each comet is located in one of these particular
        # sectors, as indicated on your note sheet:
        # Standard Mode: 2, 3, 5, 7, or 11
        # (Expert Mode: 2, 3, 5, 7, 11, 13, or 17)
        valid_comet_locations = [1, 2, 4, 6, 10]
        constraints.append(
            z3.Implies(
                X[i] == ObjectType.COMET.value,
                value_in_set_constraint(i, valid_comet_locations)
            )
        )

        # Asteroids (4 total)
        # Each asteroid is adjacent to at least one other
        # asteroid. (This means that the asteroids are
        # either in two separate pairs or in one group of
        # four.)
        constraints.append(
            z3.Implies(
                X[i] == ObjectType.ASTEROID.value,
                z3.Or(
                    X[prev_sector(i)] == ObjectType.ASTEROID.value,
                    X[next_sector(i)] == ObjectType.ASTEROID.value
                )
            )
        )

        # Gas Clouds (2 total)
        # Each gas cloud is adjacent to at least one truly
        # empty sector.
        constraints.append(
            z3.Implies(
                X[i] == ObjectType.GAS_CLOUD.value,
                z3.Or(
                    X[prev_sector(i)] == ObjectType.TRULY_EMPTY.value,
                    X[next_sector(i)] == ObjectType.TRULY_EMPTY.value,
                )
            )
        )

        # Truly Empty Sectors (2 total)
        # No extra constraints for truly empty sectors.

        # Dwarf Planets (1 total)
        # No dwarf planet is adjacent to Planet X.
        constraints.append(
            z3.Implies(
                X[i] == ObjectType.DWARF_PLANET.value,
                z3.And(
                    X[prev_sector(i)] != ObjectType.PLANET_X.value,
                    X[next_sector(i)] != ObjectType.PLANET_X.value,
                )
            )
        )

        # Planet X (1 total)
        # Planet X is not adjacent to a dwarf planet.
        # In surveys and targets, the sector containing
        # Planet X appears empty.
        constraints.append(
            z3.Implies(
                X[i] == ObjectType.PLANET_X.value,
                z3.And(
                    X[prev_sector(i)] != ObjectType.DWARF_PLANET.value,
                    X[next_sector(i)] != ObjectType.DWARF_PLANET.value,
                )
            )
        )

    return constraints, X


def generate_solar_system() -> SolarSystem:
    constraints, X = get_base_system_constraints()

    solver = z3.Solver()
    for constraint in constraints:
        solver.add(constraint)

    solver.check()
    model = solver.model()

    objects = []
    for i in range(0, NUM_SECTORS):
        object = ObjectType(model[X[i]].as_long())
        objects.append(object)

    return SolarSystem(objects)


@dataclass
class SearchResult:
    time_cost: int
    actions: list[Action]


NUM_VISIBLE_SECTORS = NUM_SECTORS / 2


def get_range_cyclic(start: int, length: int) -> Iterable[int]:
    current = start
    num_yielded = 0

    while num_yielded < length:
        yield current
        num_yielded += 1
        current = next_sector(current)


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
        Survey(survey_object, start, survey_size)
        for survey_object in surveyable_objects
        for start in get_range_cyclic(visible_range_start, 3)
    ]


def pick_survey_randomly(surveys: list[Survey]) -> Survey:
    # This is obviously complete shit and uses no information.
    return random.choice(surveys)

def pick_survey_no_repetition(surveys: list[Survey], done_surveys: set[Survey]) -> Survey:
    # It's pretty obvious that we shouldn't do the same survey twice.
    choice = None

    while choice in done_surveys:
        choice = random.choice(surveys)

    return choice

def execute_action(action: Action, solar_system: SolarSystem) -> SurveyResult:
    assert isinstance(action, Survey)
    found = 0

    for sector in get_range_cyclic(action.survey_start, action.survey_size):
        if solar_system.sector_objects[sector] == action.surveying_for:
            found += 1

    return SurveyResult(found)


def deduce_planet_x_location(constraints: list, X: list[z3.Int]) -> Optional[LocatePlanetX]:
    # We've found Planet X if among all solutions matching our information, all have the
    # same Planet X location.

    # let each X_i represent the object present in the solar system at sector i

    def model_to_planet_x_location(model: z3.Model) -> LocatePlanetX:
        for i in range(0, NUM_SECTORS):
            if model[X[i]] == ObjectType.PLANET_X.value:
                return LocatePlanetX(i,
                                     ObjectType(model[X[prev_sector(i)]].as_long()),
                                     ObjectType(model[X[next_sector(i)]].as_long()))

        raise Exception("No Planet X found in model???")

    # TODO: This is actually wrong - we don't just need a unique planet X location, we
    # also need to know what's next to it!
    possible_models = []
    possible_x_locations = []

    solver = z3.Solver()
    solver.add(*constraints)

    while True:
        if solver.check() == z3.sat:
            model = solver.model()
            possible_models.append(model)
            location = model_to_planet_x_location(model)
            possible_x_locations.append(location.sector)
            solver.add(X[location.sector] != ObjectType.PLANET_X.value)
        else:
            break

    if len(possible_models) == 1:
        return model_to_planet_x_location(possible_models[0])

    return None


def find_planet_x(solar_system: SolarSystem) -> SearchResult:
    # The strategy here is:
    # * Always survey 4 sectors (narrowest 3 cost search)
    # * Never target (because it seems too expensive compared to survey + perfect deduction)
    # * Never research (because I don't know how the research results are generated)
    # * Only locate Planet X when we're 100% sure
    time = 0
    actions: list[Action] = []
    action_set = set()
    constraints, X = get_base_system_constraints()
    visible_range_start = 0

    while True:
        # TODO: This is dumb, it should be smarter
        possible_actions = get_possible_actions(visible_range_start)

        # TODO: this is where we need a good strategy
        action = pick_survey_no_repetition(possible_actions, action_set)

        actions.append(action)
        action_set.add(action)

        if isinstance(action, Survey):
            survey_result = execute_action(action, solar_system)

            constraints.append(
                object_limit_in_range_constraint(
                    X,
                    action.surveying_for,
                    survey_result.number_found,
                    get_range_cyclic(action.survey_start, action.survey_size)
                )
            )

            # cost of 4 width survey
            # TODO: don't hardcode time values
            time += 3
            visible_range_start = (visible_range_start + 3) % NUM_SECTORS

            print(f"action: {action}, time: {time}")

        planet_x_location: Optional[LocatePlanetX] = deduce_planet_x_location(constraints, X)
        if planet_x_location is not None:
            actions.append(planet_x_location)
            time += 5
            break

    return SearchResult(time, actions)


def main():
    solar_system: SolarSystem = generate_solar_system()

    print("Board:")
    for i in range(0, NUM_SECTORS):
        print(f"{i}: {solar_system.sector_objects[i].name}")

    search_result: SearchResult = find_planet_x(solar_system)

    print(search_result.actions[-1])
    print(f"Found Planet X in:\n{search_result.time_cost}")


if __name__ == "__main__":
    z3.set_option('smt.arith.random_initial_value', True)
    z3.set_option('smt.random_seed', int(2**32 * random.random()))
    main()
