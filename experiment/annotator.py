from EBH.utility.const import vidroot
from EBH.utility.frame import DataWrapper
from experiment.visualize import plot_peaks_subtract


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
            print("\nCONFIG:", ";".join("-".join(map(str, c)) for c in xarg.items()))
            plot_peaks_subtract(self.dw, xarg["threshtop"], xarg["threshbot"], filtersize=xarg["filtersize"])
            if input("Do you accept? Y/n > ").lower() == "n":
                readarg("threshtop", xarg)
                readarg("threshbot", xarg)
                readarg("filtersize", xarg)
            else:
                break
        newpeaks = self.dw.get_peaks(xarg["threshtop"], xarg["peaksize"], args=True)
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

    def slice_peak(self, peakcenter, side):
        xarg = self.extraction_arg
        time, norm = self.dw.get_data(side, norm=True)

        hsize = xarg["peaksize"] // 2
        ptime = time[peakcenter-hsize:peakcenter+hsize]
        pdata = norm[peakcenter-hsize:peakcenter+hsize]
        return ptime, pdata


if __name__ == '__main__':
    sess = Session("Bela_le")
    print("Doing session:", sess.dw.ID)
    sess.set_extraction_args()
    # sess.setup_video()
    sess.annotate_cli(side="left")
    sess.annotate_cli(side="right")
    sess.dw.save()
