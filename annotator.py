import cv2

from EBH import vidroot
from utility.frame import DataWrapper
from visualize import plot_peaks_subtract


def readarg(name, argdict, itype=int, prompt=None):
    v = input("New {}: ".format(name) if prompt is None else prompt)
    if v:
        argdict[name] = itype(v)


class Session:

    def __init__(self, datasource, offset=0, ythresh=50, peaksize=10, filtersize=5):
        self.dw = DataWrapper(datasource)
        self.offset = offset
        self.extraction_arg = dict(
            peaksize=peaksize, filtersize=filtersize, threshtop=ythresh, threshbot=ythresh
        )
        self.extraction_arg.update(self.dw.cfg)
        self.video_arg = dict(
            dev=None, source="{}{}.avi".format(vidroot, self.dw.ID), nframes=0, vlen=0., offset=offset
        )
        self.peaks = {"left": [], "right": []}

    def set_extraction_args(self):
        xarg = self.extraction_arg
        while 1:
            print("\nCurrent config:", xarg)
            plot_peaks_subtract(self.dw, xarg["threshtop"], xarg["threshbot"], filtersize=xarg["filtersize"])
            if input("Do you accept? Y/n > ").lower() == "n":
                readarg("threshtop", xarg)
                readarg("threshbot", xarg)
                readarg("filtersize", xarg)
            else:
                break
        newpeaks = self.dw.get_peaks_subtract(xarg["threshtop"], xarg["peaksize"], args=True)
        self.peaks.update(dict(zip(("left", "right"), newpeaks)))

    def annotate_cli(self, side):
        ofs = self.video_arg["offset"]
        for p in self.peaks[side]:
            peaktime, peakdata = self.slice_peak(p, "left")
            print(f"SIDE: {side.upper()} of session: {self.dw.ID}")
            print(f"PKT: {peaktime[0]/1000:.2f} - {peaktime[-1]/1000:.2f}")
            print(f"OFF: {(ofs + peaktime[0])/1000:.2f} - {(ofs + peaktime[-1])/1000:.2f}")
            print(f"LEN: {(peaktime[-1] - peaktime[0])/1000:.4f} s")
            self.dw.annot[side[0]] = "JHU".index(input("Annotation (J/H/U): "))

    def setup_video(self):
        varg = self.video_arg
        # noinspection PyArgumentList
        dev = cv2.VideoCapture(varg["source"])
        if not dev.isOpened():
            varg["dev"] = None
            return
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
        # noinspection PyTypeChecker
        readarg("offset", varg, itype=float)

    def slice_peak(self, peakcenter, side):
        xarg = self.extraction_arg
        time, norm = self.dw.get_data(side, norm=True)

        hsize = xarg["peaksize"] // 2
        ptime = time[peakcenter-hsize:peakcenter+hsize]
        pdata = norm[peakcenter-hsize:peakcenter+hsize]
        return ptime, pdata

    def slice_video(self, ptime, pdata):
        varg = self.video_arg
        dev = varg["dev"]
        fpms = varg["nframes"] / varg["vlen"]
        nframes = (ptime[-1] - ptime[0]) / fpms
        dev.set(cv2.CAP_PROP_POS_MSEC, varg["offset"] + ptime[0])
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
    sess = Session("Virginia_le")
    print("Doing session:", sess.dw.ID)
    sess.set_extraction_args()
    # sess.setup_video()
    sess.annotate_cli(side="left")
    sess.annotate_cli(side="right")
    sess.dw.save()
