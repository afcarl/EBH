import matplotlib.pyplot as plt
import cv2

from utility.frame import DataWrapper
from utility.peak import find_peaks

from EBH import vidroot


class Peak:

    def __init__(self, time, data, videosource, offset=0):
        self.time = time
        self.data = data
        self.vsource = videosource
        self.offset = offset
        self.annotation = None

    def annotate(self):
        vdev = cv2.VideoCapture(self.vsource)
        vdev.set(cv2.CAP_PROP_POS_MSEC, self.offset)
        plt.ion()
        fig, axarr = plt.subplots(2, 1)
        axarr[1].plot(self.time, self.data)
        xobj, = axarr[1].plot(self.time[0], self.data[0], "rx")
        vobj = axarr[0].imshow(vdev.read()[-1])
        for time, data in zip(self.time, self.data):
            vdev.set(cv2.CAP_PROP_POS_MSEC, time+self.offset)
            success, frame = vdev.read()
            xobj.set_data(time, data)
            vobj.set_data(frame[:, :, ::-1])
            plt.pause(0.1)
        plt.close()
        self.annotation = "JHU".index(input("J/H/U > "))


TARGET = "Bela_fel"

dw = DataWrapper(TARGET)

ltime, ldata = dw.left

lpeakarg = find_peaks(ldata, center=False)

for s, e in lpeakarg:
    X, Y = ltime[s-5:e+5], plt.np.linalg.norm(ldata[s-5:e+5], axis=1)
    peak = Peak(X, Y, vidroot + TARGET + ".mp4", offset=2550)
    peak.annotate()
