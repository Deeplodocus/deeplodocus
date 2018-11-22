"""
Test projectStructure class
"""

from deeplodocus.core.project.project_utility import ProjectUtility

p = ProjectUtility(force_overwrite=True)
p.generate_structure()
