from deeplodocus.utils.flag import Flag

#
# COMPARISON FOR THE OVERWATCH METRIC
#
DEEP_COMPARE_METRIC_SMALLER = Flag(name="Compare Metric Smaller",
                                   description="Save if new metric smaller than the previous one",
                                   names=["default", "smaller", "<"])
DEEP_COMPARE_METRIC_BIGGER = Flag(name="Compare Metric Bigger",
                                  description="Save if new metric bigger than the previous one",
                                  names=["bigger", ">"])