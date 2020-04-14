from deeplodocus.utils.flag import Flag

DEEP_DATASET_TRAIN = Flag(
    name="Train",
    description="Training portion of the dataset",
    names=["train", "training"]
)

DEEP_DATASET_VAL = Flag(
    name="Validation",
    description="Validation portion of the dataset",
    names=["validation", "val"]
)

DEEP_DATASET_TEST = Flag(
    name="Test",
    description="Testportion of the dataset",
    names=["test", "testing"]
)

DEEP_DATASET_PREDICTION = Flag(
    name="Prediction",
    description="Prediction portion of the dataset",
    names=["predict", "prediction", "pred"]
)
