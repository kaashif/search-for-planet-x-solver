from planet_x.generation import legal_comet_sectors
from planet_x.types import *
from planet_x.board import *


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


def survey_cost(survey: Survey) -> int:
    if 1 <= survey.survey_size <= 3:
        return 4
    if 4 <= survey.survey_size <= 6:
        return 3
    else:
        raise Exception("the fuck kind of survey is this?")


def execute_action(action: Action, solar_system: SolarSystem) -> SurveyResult:
    assert isinstance(action, Survey)
    found = 0

    for sector in get_range_cyclic(action.survey_start, action.survey_size):
        if solar_system.sector_objects[sector] == action.surveying_for:
            found += 1

    return SurveyResult(found)
