
import numpy as np
import matplotlib.pyplot as plt

def get_of(mid_frame :str):
    return mid_frame.split(".")[0]+"_opticalflow_0.npy"

for annotation in ["mid_frame_test.csv", "mid_frame_train.csv", "mid_frame_val.csv"]:
    print(annotation)
    with open(annotation) as csvfile:
        for row in csvfile:
            filename, label = row.split(",")
            of_filename = get_of(filename)
            of_file = np.load("optical_flow_mid_frames/" + of_filename)
            image = plt.imread('mid_frames/' + filename)  
            image_array = np.array(image)
            stacked_array = np.dstack((image_array,of_file))
            np.save("fusion/" + filename, stacked_array)

from glob import glob

fusions = glob("fusion/*")
for fusion in fusions:
    print(np.load(fusion).shape)