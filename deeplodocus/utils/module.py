from deeplodocus.utils.notification import Notification
from deeplodocus.utils.flags.notif import *
from deeplodocus.utils.flags.msg import *


def get_module(module, name, silence=False):
    local = {"module": None}
    try:
        exec("from %s import %s\nmodule = %s" % (module, name, name), {}, local)
        Notification(DEEP_NOTIF_SUCCESS, DEEP_MSG_MODULE_LOADED % (name, module))
    except ImportError as e:
        if not silence:
            Notification(DEEP_NOTIF_ERROR, e)
    return local["module"]

