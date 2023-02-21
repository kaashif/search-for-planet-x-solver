import random
import logging

from planet_x.generation import generate_all_systems
from planet_x.main import *
from planet_x.scoring import *

logger = logging.getLogger()

def main():
    all_systems: list[SolarSystem] = generate_all_systems()

    strategies = {
        "max information content": expected_information_content_per_time,
        "max num choices per time": num_distinct_results_per_time,
        "max exp neg prob per time": negative_expected_probability_per_time,
    }

    name_to_time_costs = {
        name: [] for name in strategies.keys()
    }

    for run in range(0, 1_000_000):
        solar_system: SolarSystem = random.choice(all_systems)

        logger.debug("Board:")
        for i in range(0, NUM_SECTORS):
            logger.debug(f"{i}: {solar_system.sector_objects[i].name}")

        results = {}

        for name, evaluator in strategies.items():
            logger.debug(f"using strategy {name}")

            try:
                search_result: SearchResult = find_planet_x(solar_system, evaluator, all_systems)
            except:
                continue

            logger.debug(search_result.actions[-1])
            logger.debug(f"Found Planet X in: {search_result.time_cost} days")
            logger.debug(f"num actions = {len(search_result.actions)}\n")
            logger.debug("#" * 80)

            results[name] = search_result.time_cost

            name_to_time_costs[name].append(search_result.time_cost)

        for name, cost in results.items():
            logger.debug(f"{name} - time taken = {cost}")

        if run % 10 == 9:
            print(".", end="", flush=True)

        if run % 100 == 99:
            print("")
            for name, costs in name_to_time_costs.items():
                print(f"{name}: avg(time) = {sum(costs) / len(costs)}")

main()