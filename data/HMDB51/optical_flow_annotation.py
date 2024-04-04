import csv
from sys import argv

def annotate(original: str, target: str, intervals: int) -> None:
    with open(original) as orig, open(target, "w", newline="") as tar:
        origreader = csv.reader(orig)
        tarwriter = csv.writer(tar)
        for line in origreader:
            oldname, label = line
            for i in range(intervals):
                tarwriter.writerow([newname(oldname, i), label])

def newname(originalname: str, interval: int) -> str:
    newname = f"{originalname[:-4]}_opticalflow_{interval}.npy"
    return newname

if __name__=="__main__":
    orig = argv[1]
    new = argv[2]
    interval = int(argv[3])
    annotate(orig, new, interval)
