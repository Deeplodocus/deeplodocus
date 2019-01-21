"""
Test projectStructure class
"""
import argparse

from deeplodocus.core.project.project_utility import ProjectUtility

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", type=str, default="deeplodocus_project")
args = parser.parse_args()

p = ProjectUtility(project_name=args.name)
p.generate_structure()
