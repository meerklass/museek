import enum


class DemoEnum(enum.Enum):
    """ Only for demonstration use. """
    ASTRONAUT_RIDING_HORSE_IN_SPACE = enum.auto()
    ASTRONAUT_RIDING_HORSE_IN_SPACE_FLIPPED = enum.auto()

    CONTEXT_STORAGE_DIRECTORY = enum.auto()
    CONTEXT_FILE_NAME = enum.auto()
