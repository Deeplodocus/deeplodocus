"""
Test projectStructure class
"""

from deeplodocus.core.project.project_utility import ProjectUtility

p = ProjectUtility()
path = p.get_main_path()
p.generate_structure()