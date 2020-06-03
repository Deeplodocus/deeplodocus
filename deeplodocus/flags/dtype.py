from deeplodocus.utils.flag import Flag


#
# FLOATS
#

DEEP_DTYPE_FLOAT8 = Flag(name="float8",
                           description="Float8 format",
                           names=["float8"])

DEEP_DTYPE_FLOAT16 = Flag(name="float16",
                            description="Float16 format",
                            names=["float16"])

DEEP_DTYPE_FLOAT32 = Flag(name="float32",
                            description="Float32 format",
                            names=["float32", "float"])

DEEP_DTYPE_FLOAT64 = Flag(name="float64",
                            description="Float64 format",
                            names=["float64"])


#
# INTEGERS
#

DEEP_DTYPE_INT8 = Flag(name="int8",
                         description="Int8 format",
                         names=["int8"])

DEEP_DTYPE_INT16 = Flag(name="int16",
                          description="Int16 format",
                          names=["int16"])

DEEP_DTYPE_INT32 = Flag(name="int32",
                          description="Int32 format",
                          names=["int32", "int"])

DEEP_DTYPE_INT64 = Flag(name="int64",
                          description="Int64 format",
                          names=["int64"])


#
# UNSIGNED INTEGERS
#


DEEP_DTYPE_UINT8 = Flag(name="uint8",
                          description="Unsigned Int8 format",
                          names=["uint8"])

DEEP_DTYPE_UINT16 = Flag(name="uint16",
                           description="Unsigned Int16 format",
                           names=["uint16"])

DEEP_DTYPE_UINT32 = Flag(name="uint32",
                           description="Unsigned Int32 format",
                           names=["uint32", "uint"])

DEEP_DTYPE_UINT64 = Flag(name="uint64",
                           description="Unsigned Int64 format",
                           names=["uint64"])

#
# STRING
#

DEEP_DTYPE_STR = Flag(name="str",
                        description="String format",
                        names=["str"])
