from glob import glob
from numpy import load, max, mean, var

ofs = glob("optical_flow/*")

for of in ofs:
    array = load(of)
    print(array.shape)
    print(max(array), mean(array), var(array))
