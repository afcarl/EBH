import os
import pickle
import shutil

from EBH import projectroot, rawroot, logroot, vidroot


IDz = {}

for dirnm in os.listdir(rawroot):
    splitted = dirnm.split("_")
    ID = "_".join((splitted[0], splitted[-1]))
    os.chdir(rawroot + dirnm)
    vid = [flnm for flnm in os.listdir(".") if "mp4" == flnm[-3:]][0]
    txt = [flnm for flnm in os.listdir(".") if "txt" == flnm[-3:]][0]

    viddest = vidroot + f"{ID}.mp4"
    txtdest = logroot + f"{ID}.txt"

    IDz[ID] = (txtdest, viddest)

    shutil.move(vid, viddest)
    shutil.move(txt, txtdest)

with open(projectroot + "IDz.pkl", "wb") as handle:
    pickle.dump(IDz, handle)
