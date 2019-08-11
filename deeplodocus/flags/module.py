import torch
import torch.nn.functional

from deeplodocus.utils import get_main_path
import deeplodocus.app.transforms as deep_tfm
import deeplodocus.app.optimizers as deep_optim
import deeplodocus.app.models as deep_models
import deeplodocus.app.losses as deep_losses
import deeplodocus.app.metrics as deep_metrics

import torchvision.datasets as tv_data

DEEP_MODULE_OPTIMIZERS = {"pytorch": {"path": torch.optim.__path__,
                                      "prefix": torch.optim.__name__},
                          "deeplodocus": {"path": deep_optim.__path__,
                                          "prefix":deep_optim.__name__},
                          "custom": {"path": [get_main_path() + "/modules/optimizers"],
                                     "prefix": "modules.optimizers"}
                          }

DEEP_MODULE_MODELS = {"deeplodocus": {"path": deep_models.__path__,
                                      "prefix":deep_models.__name__},
                      "custom": {"path": ["%s/modules/models" % get_main_path()],
                                 "prefix": "modules.models"}
                      }

DEEP_MODULE_LOSSES = {"pytorch": {"path": torch.nn.__path__,
                                  "prefix": torch.nn.__name__},
                      "deeplodocus": {"path": deep_losses.__path__,
                                      "prefix":deep_losses.__name__},
                      "custom": {"path": [get_main_path() + "/modules/losses"],
                                 "prefix": "modules.losses"}
                      }

DEEP_MODULE_METRICS = {"pytorch": {"path": torch.nn.__path__,
                                   "prefix": torch.nn.__name__},
                       "deeplodocus_m": {"path": deep_metrics.__path__, # key has to be different from the one in the losses
                                       "prefix": deep_metrics.__name__},
                       "custom_m": {"path": [get_main_path() + "/modules/metrics"], # key has to be different from the one in the losses
                                  "prefix": "modules.metrics"}
                       }

DEEP_MODULE_TRANSFORMS = {"deeplodocus": {"path": deep_tfm.__path__,
                                          "prefix": deep_tfm.__name__},
                          "custom": {"path": [get_main_path() + "/modules/transforms"],
                                     "prefix": "modules.transforms"}
                          }


DEEP_MODULE_DATASETS = {"torchvision": {"path": tv_data.__path__,
                                          "prefix": tv_data.__name__},

                        "custom": {"path": [get_main_path() + "/modules/datasets"],
                                   "prefix": "modules.datasets"}
                        }
