import time
import logging

from z3 import z3
from planet_x.types import *
from planet_x.board import *

logger = logging.getLogger()

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
    logger.debug(f"generating models took {end - start} seconds")
    return models


def model_to_system(model: z3.ModelRef, X) -> SolarSystem:
    objects = []
    for i in range(0, NUM_SECTORS):
        object = ObjectType(model[X[i]].as_long())
        objects.append(object)

    return SolarSystem(objects)


def generate_all_systems() -> list[SolarSystem]:
    constraints, X = get_base_system_constraints()

    return [
        model_to_system(model, X)
        for model in get_all_possible_models(constraints, X)
    ]
