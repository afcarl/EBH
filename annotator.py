from types import SimpleNamespace

import cv2
import matplotlib.pyplot as plt

from visualize import plot_peaks_subtract
from utility.frame import DataWrapper

from EBH import vidroot


class Session:

    def __init__(self, datasource, offset=0, ythresh=50, peaksize=10, filtersize=5):
        self.dw = DataWrapper(datasource)
        self.offset = offset
        self.extraction_arg = SimpleNamespace(peaksize=peaksize, filtersize=filtersize, threshold=ythresh)
        self.video_arg = SimpleNamespace(
            dev=None, source=f"{vidroot}{self.dw.ID}.mp4", nframes=0, vlen=0., offset=offset
        )
        self.peaks = None

    def extract_peaks(self):
        xarg = self.extraction_arg
        while 1:
            print(f"\nCurrent config: threshold: {xarg.threshold]} filtersize: {xarg.filtersize}")
            plot_peaks_subtract(self.dw, xarg.threshold, xarg.filtersize)
            if input("Do you accept? Y/n > ").lower() == "y":
                break
            xarg.treshold = int(input("New threshold: "))
            xarg.filtersize = int(input("New filtersize: "))
        self.peaks = self.dw.get_peaks_subtract(xarg.threshold, xarg.peaksize)

    def setup_video(self):
        varg = self.video_arg
        # noinspection PyArgumentList
        dev = cv2.VideoCapture(varg.source)
        succ, frame = dev.read()
        assert succ
        dev.set(cv2.CAP_PROP_POS_AVI_RATIO, 1.)
        varg.nframes = dev.get(cv2.CAP_PROP_POS_FRAMES)
        varg.vlen = dev.get(cv2.CAP_PROP_POS_MSEC)
        varg.dev = dev
        dev.set(cv2.CAP_PROP_POS_AVI_RATIO, 0.)
        for frno in range(varg.nframes):
            succ, frame = dev.read()
            assert succ
            cv2.imshow(self.dw.ID, frame)
            key = cv2.waitKey(100)
            if key == 32:
                while not key:
                    key = cv2.waitKey(1000)
            if key == 27:
                break
        varg.offset = float(input("Delay as float: "))

    def slice_peak(self, peakcenter, side):
        time = self.dw.get

    def peak_replay(self):
        lastime = max(cv2)
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
