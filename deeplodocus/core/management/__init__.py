from deeplodocus.core.management.management_utility import ManagementUtility


def execute_from_command_line(argv=None):
    """Run a ManagementUtility."""
    utility = ManagementUtility(argv)
    utility.execute_from_command_line()
