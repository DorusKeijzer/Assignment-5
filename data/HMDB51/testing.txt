from glob import glob
from numpy import load
ofs = glob("optical_flow/*")

for of in ofs:
    array = load(of)
    print(array.shape)