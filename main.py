import math
import random
import time
from copy import deepcopy

from z3 import z3
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Iterable, Callable

verbose = True


def print_v(s):
    if verbose:
        print_v(s)


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
    return sum(z3.If(X[i] == obj_type.value, 1, 0) for i in sectors) == count


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

legal_comet_sectors = [1, 2, 4, 6, 10]


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
        constraints.append(
            z3.Implies(
                X[i] == ObjectType.COMET.value,
                value_in_set_constraint(i, legal_comet_sectors)
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


def generate_all_systems() -> list[SolarSystem]:
    constraints, X = get_base_system_constraints()

    return [
        model_to_system(model, X)
        for model in get_all_possible_models(constraints, X)
    ]


@dataclass
class SearchResult:
    time_cost: int
    actions: list[Action]


NUM_VISIBLE_SECTORS = NUM_SECTORS // 2


def get_range_cyclic(start: int, length: int) -> Iterable[int]:
    current = start
    num_yielded = 0

    while num_yielded < length:
        yield current
        num_yielded += 1
        current = next_sector(current)


def get_all_possible_models(constraints: list, X: list[z3.BitVec]) -> list[z3.ModelRef]:
    start = time.time()
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

    end = time.time()
    print_v(f"generating models took {end - start} seconds")
    return models


def get_possible_actions(visible_range_start: int) -> list[Action]:
    # You can't survey for Planet X
    surveyable_objects = set(ObjectType)
    surveyable_objects.remove(ObjectType.PLANET_X)

    survey_sizes = range(2, NUM_VISIBLE_SECTORS + 1)

    surveys = []

    for survey_size in survey_sizes:
        num_start_positions = NUM_VISIBLE_SECTORS - survey_size + 1
        surveys += [
            Survey(survey_object, start, survey_size)
            for survey_object in surveyable_objects
            for start in get_range_cyclic(visible_range_start, num_start_positions)
        ]

    # Search for a comet must start and end on a legal comet sector
    legal_surveys = [
        survey
        for survey in surveys
        if survey.surveying_for != ObjectType.COMET
           or (survey.survey_start in legal_comet_sectors
               and ((survey.survey_start + survey.survey_size - 1) % NUM_SECTORS) in legal_comet_sectors)
    ]

    return legal_surveys


def model_to_system(model: z3.ModelRef, X) -> SolarSystem:
    objects = []
    for i in range(0, NUM_SECTORS):
        object = ObjectType(model[X[i]].as_long())
        objects.append(object)

    return SolarSystem(objects)


def survey_cost(survey: Survey) -> int:
    if 1 <= survey.survey_size <= 3:
        return 4
    if 4 <= survey.survey_size <= 6:
        return 3
    else:
        raise Exception("the fuck kind of survey is this?")


def system_by_survey_result(survey: Survey, systems: list[SolarSystem]) -> dict[int, list[SolarSystem]]:
    result_to_systems = {}

    for system in systems:
        result = execute_action(survey, system).number_found

        if result in result_to_systems:
            result_to_systems[result].append(system)
        else:
            result_to_systems[result] = [system]

    return result_to_systems


def expected_information_content_per_time(survey: Survey, systems: list[SolarSystem]) -> float:
    result_to_systems = system_by_survey_result(survey, systems)
    num_systems = len(systems)

    expected_information_content = -1 * sum(
        len(systems_with_result) * (math.log2(len(systems_with_result)) - math.log2(num_systems))
        for systems_with_result in result_to_systems.values()
    ) / num_systems
    time_cost = survey_cost(survey)

    return survey, \
        expected_information_content / time_cost, \
        f"{expected_information_content=}\n{time_cost=}"


def num_distinct_results_per_time(survey: Survey, systems: list[z3.ModelRef]) -> float:
    distinct_results = set()

    for system in systems:
        result = execute_action(survey, system).number_found
        distinct_results.add(result)

    return survey, len(distinct_results)/survey_cost(survey), distinct_results


def pick_best_survey(surveys: list[Survey],
                     done_surveys: set[Survey],
                     found_objects: set[ObjectType],
                     possible_systems: list[SolarSystem],
                     evaluator) -> Survey:
    start = time.time()
    # It's pretty obvious that we shouldn't do the same survey twice.
    # We also shouldn't survey for objects we've already found
    possible_surveys = [
        survey for survey in surveys
        if survey not in done_surveys
           and survey.surveying_for not in found_objects
    ]

    evaluations = [
        evaluator(survey, possible_systems) for survey in possible_surveys
    ]

    best_survey, score, other = max(evaluations, key=lambda e: e[1])
    print_v(other)

    end = time.time()
    print_v(f"picking best survey took {end - start} seconds")
    return best_survey


def execute_action(action: Action, solar_system: SolarSystem) -> SurveyResult:
    assert isinstance(action, Survey)
    found = 0

    for sector in get_range_cyclic(action.survey_start, action.survey_size):
        if solar_system.sector_objects[sector] == action.surveying_for:
            found += 1

    return SurveyResult(found)


def deduce_planet_x_location(possible_systems: list[SolarSystem]) -> Optional[LocatePlanetX]:
    print_v(f"num possible systems = {len(possible_systems)}")

    def system_to_planet_x_location(system: SolarSystem) -> LocatePlanetX:
        for i in range(0, NUM_SECTORS):
            if system.sector_objects[i] == ObjectType.PLANET_X:
                return LocatePlanetX(i,
                                     system.sector_objects[prev_sector(i)],
                                     system.sector_objects[next_sector(i)])

        raise Exception("No Planet X found in model???")

    possible_x_locations = {i for system in possible_systems for i in range(0, NUM_SECTORS) if
                            system.sector_objects[i] == ObjectType.PLANET_X}

    if len(possible_x_locations) == 1:
        # We've narrowed the location of Planet X down to one place, but we
        # may still need to determine what's next to Planet X.
        x_location = system_to_planet_x_location(possible_systems[0])

        p = prev_sector(x_location.sector)
        n = next_sector(x_location.sector)

        know_prev_sector = len(set(system.sector_objects[p] for system in possible_systems)) == 1
        know_next_sector = len(set(system.sector_objects[n] for system in possible_systems)) == 1

        if know_prev_sector and know_next_sector:
            # We know where X is and what's next to X!
            return x_location

    return None


def find_planet_x(solar_system: SolarSystem, survey_evaluator: Callable,
                  all_systems: list[SolarSystem]) -> SearchResult:
    # The strategy here is:
    # * Always survey 4 sectors (narrowest 3 cost search)
    # * Never target (because it seems too expensive compared to survey + perfect deduction)
    # * Never research (because I don't know how the research results are generated)
    # * Only locate Planet X when we're 100% sure
    time = 0
    visible_range_start = 0
    actions: list[Action] = []
    action_set = set()

    possible_systems = deepcopy(all_systems)

    while True:
        object_to_possible_locations = {
            object_type: set()
            for object_type in ObjectType
        }
        for system in possible_systems:
            for i in range(0, NUM_SECTORS):
                object_type = system.sector_objects[i]
                object_to_possible_locations[object_type].add(i)

        # Objects we've fully determined the positions of, no need to survey for those.
        # This is really just a performance optimization, any sensible strategy would rank these surveys right
        # at the bottom.
        found_objects = set(object_type
                            for object_type in ObjectType
                            if len(object_to_possible_locations[object_type]) == object_to_num_extant[object_type])

        planet_x_location: Optional[LocatePlanetX] = deduce_planet_x_location(possible_systems)

        if planet_x_location is not None:
            actions.append(planet_x_location)
            time += 5
            break

        possible_actions = get_possible_actions(visible_range_start)
        print_v(f"possible actions: {len(possible_actions)}")

        best_survey = pick_best_survey(
            possible_actions, action_set, found_objects, possible_systems, survey_evaluator
        )

        actions.append(best_survey)
        action_set.add(best_survey)

        possible_results_to_systems = system_by_survey_result(best_survey, possible_systems)
        num_found = execute_action(best_survey, solar_system).number_found
        prob_of_that_result = len(possible_results_to_systems[num_found]) / len(possible_systems)
        actual_information_content = -1 * math.log2(prob_of_that_result)

        possible_systems = possible_results_to_systems[num_found]

        cost = survey_cost(best_survey)
        time += cost
        visible_range_start = (visible_range_start + cost) % NUM_SECTORS

        print_v(f"action: {best_survey}, time: {time}")
        print_v(f"{actual_information_content=}")
        print_v("")

    return SearchResult(time, actions)


def match_clue(system: SolarSystem) -> bool:
    for i in range(0, NUM_SECTORS):
        if system.sector_objects[i] == ObjectType.ASTEROID and (
                ObjectType.DWARF_PLANET in [system.sector_objects[prev_sector(i)],
                                            system.sector_objects[next_sector(i)]]):
            return False
    return True


def main():
    all_systems: list[SolarSystem] = generate_all_systems()

    strategies = {
        "max information content": expected_information_content_per_time,
        "max num choices per time": num_distinct_results_per_time,
    }

    name_to_time_costs = {
        name: [] for name in strategies.keys()
    }

    for run in range(0, 1_000_000):
        solar_system: SolarSystem = random.choice(all_systems)

        print_v("Board:")
        for i in range(0, NUM_SECTORS):
            print_v(f"{i}: {solar_system.sector_objects[i].name}")

        results = {}

        for name, evaluator in strategies.items():
            print_v(f"using strategy {name}")

            try:
                search_result: SearchResult = find_planet_x(solar_system, evaluator, all_systems)
            except:
                continue

            print_v(search_result.actions[-1])
            print_v(f"Found Planet X in: {search_result.time_cost} days")
            print_v(f"num actions = {len(search_result.actions)}\n")
            print_v("#" * 80)

            results[name] = search_result.time_cost

            name_to_time_costs[name].append(search_result.time_cost)

        for name, cost in results.items():
            print_v(f"{name} - time taken = {cost}")

        if run % 10 == 9:
            print(".", end="", flush=True)

        if run % 100 == 99:
            print("")
            for name, costs in name_to_time_costs.items():
                print(f"{name}: avg(time) = {sum(costs) / len(costs)}")


if __name__ == "__main__":
    verbose = False
    main()
