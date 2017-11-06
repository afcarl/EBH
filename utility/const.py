import numpy as np
import os

projectroot = os.path.expanduser("~/SciProjects/Project_EBH/")
logroot = projectroot + "log/"
rawroot = projectroot + "raw/"
vidroot = projectroot + "vid/"
labroot = projectroot + "man/"
pklroot = projectroot + "pkl/"

labels = "?C0JUH"
dummy = tuple(range(len(labels)))
onehot = np.eye(len(dummy))
