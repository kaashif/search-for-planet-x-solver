import math
import time
import logging
from copy import deepcopy

from typing import Optional, Callable

from planet_x.generation import object_to_num_extant
from planet_x.scoring import system_by_survey_result
from planet_x.actions import *

logger = logging.getLogger()

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
    logger.debug(other)

    end = time.time()
    logger.debug(f"picking best survey took {end - start} seconds")
    return best_survey


def deduce_planet_x_location(possible_systems: list[SolarSystem]) -> Optional[LocatePlanetX]:
    logger.debug(f"num possible systems = {len(possible_systems)}")

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
        logger.debug(f"possible actions: {len(possible_actions)}")

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

        logger.debug(f"action: {best_survey}, time: {time}")
        logger.debug(f"{actual_information_content=}")
        logger.debug("")

    return SearchResult(time, actions)
