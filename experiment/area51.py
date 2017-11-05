from EBH.utility.frame import DataWrapper


dw = DataWrapper("box4_le")
lX, rX = dw.get_peaks(peaksize=10)
lY, rY = dw.get_annotations("left"), dw.get_annotations("right")

print("Data shapes in", dw.ID)
print("lX:", lX.shape)
print("lY:", lY.shape)
print("rX:", rX.shape)
print("rY:", rY.shape)
