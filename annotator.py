import cv2
import matplotlib.pyplot as plt

from utility.frame import DataWrapper

from EBH import vidroot


class Session:

    def __init__(self, datasource, offset=0, ythresh=50, peaksize=10):
        self.dw = DataWrapper(datasource)
        self.vsource = f"{vidroot}{self.dw.ID}.mp4"
        self.offset = offset
        self.threshold = ythresh
        self.peaksize = peaksize

    def reread_params(self):


    def peak_replay(self, side):
        # noinspection PyArgumentList
        vdev = cv2.VideoCapture(self.vsource)
        vdev.set(cv2.CAP_PROP_POS_MSEC, self.offset)
        plt.ion()
        fig, axarr = plt.subplots(2, 1)
        time, data = self.dw.get_data(side, norm=True)
        axarr[1].plot(time, data)
        xobj, = axarr[1].plot(time[0], data[0], "rx")
        vobj = axarr[0].imshow(vdev.read()[-1])
        for time, data in zip(time, data):
            vdev.set(cv2.CAP_PROP_POS_MSEC, time+self.offset)
            success, frame = vdev.read()
            xobj.set_data(time, data)
            vobj.set_data(frame[:, :, ::-1])
            plt.pause(0.1)
        plt.close()
