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


def calculate_optical_flow(video_path: str, intervals: int) -> None:
    # Open the video file
    cap = cv.VideoCapture(video_path)
    flows = []

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
            flows.append(cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0))
            interval_no+=1
        
        if interval_no == intervals:
            break
        frame_no += 1

        # Update the previous frame
        prvs = next

    cap.release()
    return flows

def filename(inputstring: str) -> str:
    string = inputstring[7:]
    string = string.split(".")[0]
    return string


if __name__ == "__main__":
    import os
    INTERVALS_PER_VIDEO = 4
    output_directory = "of_stacks"
    os.makedirs(output_directory, exist_ok=True)

    files = glob("videos/*")
    for _, file in enumerate(files):
        print(file)
        frames = calculate_optical_flow(file, INTERVALS_PER_VIDEO)

        reshaped_frames = [frame.reshape((*frame.shape[:-1], 1, frame.shape[-1])) for frame in frames]

        # Stack the arrays along the third axis
        stacked_array = np.concatenate(reshaped_frames, axis=2)

        # Reshape the stacked array to have shape (240, 320, 8)
        stacked_array = stacked_array.reshape((*stacked_array.shape[:2], -1))
        print(stacked_array.shape)
        np.save(os.path.join(output_directory, f"{filename(file)}.npy"), stacked_array)
