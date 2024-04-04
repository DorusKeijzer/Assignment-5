import cv2 as cv
from glob import glob
import numpy as np
from math import ceil

def split_indices(length: int, intervals: int):
    """Divides the input into the given number of intervals excluding the first frame. 
        If intervals is 1, take the mid point"""
    if intervals == 1:
        return [length//2]
    section_size = length // intervals
    indices = [section_size * i for i in range(1, intervals)]
    indices.append(length)
    return indices


def get_output(video_path:str, frame_no: int)-> str:
    filename = video_path.split("\\")[-1]
    output = "optical_flow/" + filename[:-4] + "_opticalflow_" +str(frame_no) + ".npy"
    return output

def calculate_optical_flow(video_path: str, intervals: int) -> None:
    # Open the video file
    cap = cv.VideoCapture(video_path)
    filenames = []

    # Initialize variables for the first frame
    ret, frame1 = cap.read()
    prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255

    length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    split = split_indices(length, intervals+1) 
    frame_no = 0
    interval_no = 0

    while(cap.isOpened()):
        # read the current image
        ret, frame2 = cap.read()
        if not ret:
            print("not ret")
            break
        next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        if frame_no in split:
            # Calculate the optical flow using Lucas-Kanade method
            flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            filename = get_output(video_path, interval_no)
            np.save(filename, flow)
            interval_no+=1
        
        if interval_no == intervals:
            break
        frame_no += 1

        # Update the previous frame
        prvs = next

    cap.release()


if __name__ == "__main__":
    INTERVALS_PER_VIDEO = 4

    files = glob("videos/*")
    for i, file in enumerate(files):
        print(file)
        filenames = calculate_optical_flow(file, INTERVALS_PER_VIDEO)
