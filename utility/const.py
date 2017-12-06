import numpy as np
import os

projectroot = os.path.expanduser("~/SciProjects/Project_EBH/")
logroot = projectroot + "log/"
labroot = projectroot + "man/"
ltbroot = projectroot + "ltb/"

labels = "JUHC0"
DEFAULT_DATASET = ltbroot + "default.pkl.gz"
boxer_names = ("Anita", "Bela", "box1", "box4", "box5", "box6", "David", "Dia", "Robi", "Szilard", "Toni", "Virginia")
professionals = ("box1", "box4", "box5", "box6", "David", "Dia", "Virginia")
amateurs = ("Anita", "Bela", "Robi", "Szilard", "Toni")
