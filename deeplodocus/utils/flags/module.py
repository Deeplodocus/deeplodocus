import torch
import torch.nn.functional

from deeplodocus.utils import get_main_path
import deeplodocus.data.transforms as tfm


DEEP_MODULE_OPTIMIZERS = {"pytorch":
                             {"path" : torch.optim.__path__,
                              "prefix" : torch.optim.__name__},
                          "custom":
                             {"path": [get_main_path() + "/modules/optimizers"],
                              "prefix": "modules.optimizers"}
                          }

DEEP_MODULE_MODELS = {"custom":
                          {"path": [get_main_path() + "/modules/models"],
                           "prefix": "modules.models"}
                      }

DEEP_MODULE_LOSSES = {"pytorch":
                             {"path" : torch.nn.__path__,
                              "prefix" : torch.nn.__name__},
                      "custom":
                             {"path": [get_main_path() + "/modules/losses"],
                              "prefix": "modules.losses"}
                      }

DEEP_MODULE_METRICS = {"pytorch":
                             {"path" : torch.nn.__path__,
                              "prefix" : torch.nn.__name__},
                       "custom":
                             {"path": [get_main_path() + "/modules/metrics"],
                              "prefix": "modules.metrics"}
                       }

DEEP_MODULE_TRANSFORMS = {"deeplodocus":
                             {"path" : tfm.__path__,
                              "prefix" : tfm.__name__},
                          "custom":
                                 {"path": [get_main_path() + "/modules/transforms"],
                                  "prefix": "modules.transforms"}
                          }