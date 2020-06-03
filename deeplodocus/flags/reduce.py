from deeplodocus.utils.flag import Flag


DEEP_REDUCE_MEAN = Flag(
    name="Mean",
    description="Reduction by mean",
    names=["mean"]
)

DEEP_REDUCE_SUM = Flag(
    name="Sum",
    description="Reduction by sum",
    names=["sum"]
)

DEEP_REDUCE_LAST = Flag(
    name="Last",
    description="Reduction by taking last value",
    names=["last", "latest"]
)
