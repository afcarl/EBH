import numpy as np
import os

projectroot = os.path.expanduser("~/SciProjects/Project_EBH/")
logroot = projectroot + "log/"
rawroot = projectroot + "raw/"
vidroot = projectroot + "vid/"
labroot = projectroot + "man/"
pklroot = projectroot + "pkl/"

labels = "?C0JUH"
dummy = (0, 1, 2, 3, 4)
onehot = np.eye(len(dummy))
