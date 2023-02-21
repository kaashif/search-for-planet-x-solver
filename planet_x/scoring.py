import math

from planet_x.actions import survey_cost, execute_action
from planet_x.types import *

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


def num_distinct_results_per_time(survey: Survey, systems: list[SolarSystem]) -> float:
    distinct_results = set()

    for system in systems:
        result = execute_action(survey, system).number_found
        distinct_results.add(result)

    return survey, len(distinct_results) / survey_cost(survey), distinct_results


def negative_expected_probability_per_time(survey, systems):
    result_to_systems = system_by_survey_result(survey, systems)
    num_systems = len(systems)
    time_cost = survey_cost(survey)

    neg_exp_prob = -1 * sum(
        len(systems_with_result) ** 2
        for systems_with_result in result_to_systems.values()
    ) / (num_systems ** 2)

    return survey, neg_exp_prob / time_cost, f"{neg_exp_prob=}, {time_cost=}"
