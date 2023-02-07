import math
import random
from copy import deepcopy
from functools import reduce
from operator import mul

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


object_to_num_extant = {
    ObjectType.PLANET_X: 1,
    ObjectType.DWARF_PLANET: 1,
    ObjectType.GAS_CLOUD: 2,
    ObjectType.TRULY_EMPTY: 2,
    ObjectType.COMET: 2,
    ObjectType.ASTEROID: 4
}


def get_base_system_constraints():
    X = [z3.BitVec(f"X_{i}", 4) for i in range(0, NUM_SECTORS)]
    constraints = []

    # There's only one global constraint: the numbers of objects.
    constraints += [
        total_object_limit_constraint(X, object_type, num)
        for object_type, num in object_to_num_extant.items()
    ]

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

    return model_to_system(model, X)


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


def get_all_possible_models(constraints: list, X: list[z3.BitVec]) -> list[z3.ModelRef]:
    models = []
    solver = z3.Solver()
    solver.add(*constraints)

    while solver.check() == z3.sat:
        model = solver.model()
        models.append(model)

        # At least one of the variables is different
        solver.add(
            z3.Or([
                x != model[x]
                for x in X
            ])
        )

    return models


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


def model_to_system(model: z3.ModelRef, X) -> SolarSystem:
    objects = []
    for i in range(0, NUM_SECTORS):
        object = ObjectType(model[X[i]].as_long())
        objects.append(object)

    return SolarSystem(objects)


def expected_information_content(survey: Survey, models: list[z3.ModelRef], X) -> float:
    result_to_num = {}

    for model in models:
        system = model_to_system(model, X)
        result = execute_action(survey, system).number_found

        result_to_num[result] = result_to_num.get(result, 0) + 1

    N = len(result_to_num.keys())

    information_content = -1 * math.log10(reduce(mul, [num/N for num in result_to_num.values()]))/N

    return survey, information_content, result_to_num

def num_distinct_results(survey: Survey, models: list[z3.ModelRef], X) -> float:
    distinct_results = set()

    for model in models:
        system = model_to_system(model, X)
        result = execute_action(survey, system).number_found
        distinct_results.add(result)

    return survey, len(distinct_results), distinct_results

def pick_best_survey(surveys: list[Survey],
                     done_surveys: set[Survey],
                     found_objects: set[ObjectType],
                     models: list[z3.ModelRef],
                     X,
                     evaluator) -> Survey:
    # It's pretty obvious that we shouldn't do the same survey twice.
    # We also shouldn't survey for objects we've already found
    possible_surveys = [
        survey for survey in surveys
        if survey not in done_surveys
           and survey.surveying_for not in found_objects
    ]

    evaluations = [
        evaluator(survey, models, X) for survey in surveys
    ]

    best_survey, score, other = max(evaluations, key=lambda e: e[1])
    print(other)
    return best_survey


def execute_action(action: Action, solar_system: SolarSystem) -> SurveyResult:
    assert isinstance(action, Survey)
    found = 0

    for sector in get_range_cyclic(action.survey_start, action.survey_size):
        if solar_system.sector_objects[sector] == action.surveying_for:
            found += 1

    return SurveyResult(found)


def deduce_planet_x_location(models: list, X: list[z3.Int]) -> Optional[LocatePlanetX]:
    print(f"num possible models = {len(models)}")

    def model_to_planet_x_location(model: z3.Model) -> LocatePlanetX:
        for i in range(0, NUM_SECTORS):
            if model[X[i]] == ObjectType.PLANET_X.value:
                return LocatePlanetX(i,
                                     ObjectType(model[X[prev_sector(i)]].as_long()),
                                     ObjectType(model[X[next_sector(i)]].as_long()))

        raise Exception("No Planet X found in model???")

    possible_x_locations = {i for model in models for i in range(0, NUM_SECTORS) if
                            model[X[i]] == ObjectType.PLANET_X.value}

    if len(possible_x_locations) == 1:
        # We've narrowed the location of Planet X down to one place, but we
        # may still need to determine what's next to Planet X.
        x_location = model_to_planet_x_location(models[0])

        p = prev_sector(x_location.sector)
        n = next_sector(x_location.sector)

        know_prev_sector = len(set(model[X[p]].as_long() for model in models)) == 1
        know_next_sector = len(set(model[X[n]].as_long() for model in models)) == 1

        if know_prev_sector and know_next_sector:
            # We know where X is and what's next to X!
            return x_location

    return None


def find_planet_x(solar_system: SolarSystem, survey_evaluator) -> SearchResult:
    # The strategy here is:
    # * Always survey 4 sectors (narrowest 3 cost search)
    # * Never target (because it seems too expensive compared to survey + perfect deduction)
    # * Never research (because I don't know how the research results are generated)
    # * Only locate Planet X when we're 100% sure
    time = 0
    visible_range_start = 0
    actions: list[Action] = []

    action_set = set()
    constraints, X = get_base_system_constraints()

    while True:
        models = get_all_possible_models(constraints, X)
        object_to_possible_locations = {
            object_type: set()
            for object_type in ObjectType
        }
        for model in models:
            for i in range(0, NUM_SECTORS):
                object_type = ObjectType(model[X[i]].as_long())
                object_to_possible_locations[object_type].add(i)

        # Objects we've fully determined the positions of, no need to survey for those
        found_objects = set(object_type
                            for object_type in ObjectType
                            if len(object_to_possible_locations[object_type]) == object_to_num_extant[object_type])

        planet_x_location: Optional[LocatePlanetX] = deduce_planet_x_location(models, X)

        if planet_x_location is not None:
            actions.append(planet_x_location)
            time += 5
            break

        possible_actions = get_possible_actions(visible_range_start)

        # TODO: this is where we need a good strategy
        action = pick_best_survey(
            possible_actions, action_set, found_objects, models, X, survey_evaluator
        )

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

    return SearchResult(time, actions)


def main():
    solar_system: SolarSystem = generate_solar_system()

    print("Board:")
    for i in range(0, NUM_SECTORS):
        print(f"{i}: {solar_system.sector_objects[i].name}")

    results = {}

    for name, evaluator in [
        ("max information content", expected_information_content),
        ("most choices", num_distinct_results)
    ]:
        print(f"using strategy {name}")
        search_result: SearchResult = find_planet_x(solar_system, evaluator)

        print(search_result.actions[-1])
        print(f"Found Planet X in:\n{search_result.time_cost}")

        results[name] = search_result.time_cost

    for name, cost in results.items():
        print(f"{name} - cost = {cost}")

if __name__ == "__main__":
    z3.set_option('smt.arith.random_initial_value', True)
    z3.set_option('smt.random_seed', int(2 ** 32 * random.random()))
    main()
