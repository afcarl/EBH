from utility.frame import DataWrapper


dw = DataWrapper("Bela_le")
lX, rX = dw.get_peaks()
lY, rY = dw.get_annotations("left"), dw.get_annotations("right")

print("Data shapes in", dw.ID)
print("lX:", lX.shape)
print("rX:", rX.shape)
print("lY:", lY.shape)
print("rY:", rY.shape)
