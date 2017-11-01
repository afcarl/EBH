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
        self.extraction_arg = SimpleNamespace(
            peaksize=peaksize, filtersize=filtersize, threshold=ythresh, thresh2=None
        )
        self.video_arg = SimpleNamespace(
            dev=None, source="{}{}.mp4".format(vidroot, self.dw.ID), nframes=0, vlen=0., offset=offset
        )
        self.peaks = {"left": [], "right": []}

    def extract_peaks(self):
        xarg = self.extraction_arg
        while 1:
            print("\nCurrent config: threshold: {} filtersize: {}".format(xarg.threshold, xarg.filtersize))
            plot_peaks_subtract(self.dw, xarg.threshold, xarg.filtersize)
            if input("Do you accept? Y/n > ").lower() == "y":
                break
            xarg.treshold = int(input("New threshold: "))
            xarg.thresh2 = input("(New thres2): ")
            xarg.thresh2 = None if not xarg.thresh2 else int(xarg.thresh2)
            xarg.filtersize = int(input("New filtersize: "))
        self.peaks.update(dict(zip(("left", "right"), self.dw.get_peaks_subtract(xarg.threshold, xarg.peaksize))))

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
        xarg = self.extraction_arg
        varg = self.video_arg
        time, norm = self.dw.get_data(side, norm=True)

        hsize = xarg.peaksize // 2
        ptime = time[peakcenter-hsize:peakcenter+hsize]
        pdata = norm[peakcenter-hsize:peakcenter+hsize]

        dev = varg.dev
        fpms = varg.nframes / varg.vlen
        nframes = (ptime[-1] - ptime[0]) / fpms
        dev.set(cv2.CAP_PROP_POS_MSEC, varg.offset + ptime[0])
        frames = []
        for succ, frame in (dev.read() for _ in range(nframes)):
            assert succ
            frames.append(frame)
        print("frames: {} vs {} measurements".format(len(frames), len(pdata)))
        done = False
        while not done:
            for frame in frames:
                cv2.imshow(self.dw.ID, frame)
                key = cv2.waitKey(1./fpms)
                if key == 27:
                    done = True


if __name__ == '__main__':
    sess = Session("Bela_fel")
    sess.extract_peaks()
    sess.setup_video()
    for p in sess.peaks["left"]:
        sess.slice_peak(p, side="left")
