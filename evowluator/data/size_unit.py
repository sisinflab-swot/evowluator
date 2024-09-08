from pyutils.types.strenum import StrEnum


class SizeUnit(StrEnum):
    SIZE = 'size'
    AXIOMS = 'axioms'
    CLASSES = 'classes'
    DATATYPES = 'datatypes'
    NAMED_INDIVIDUALS = 'named_individuals'
    ANONYMOUS_INDIVIDUALS = 'anonymous_individuals'
    OBJECT_PROPERTIES = 'object_properties'
    DATA_PROPERTIES = 'data_properties'
    ANNOTATION_PROPERTIES = 'annotation_properties'
    PRIMITIVES = 'primitives'
    ENTITIES = 'entities'
    INDIVIDUALS = 'individuals'
    LOGICAL_PROPERTIES = 'logical_properties'
    PROPERTIES = 'properties'
