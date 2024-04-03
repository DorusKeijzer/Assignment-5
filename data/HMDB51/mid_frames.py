import cv2 as cv
from glob import glob
import numpy as np

def get_output(video_path:str)-> str:
    filename = video_path.split("\\")[-1]
    output = "mid_frames/" + filename[:-4] + "midframe.png"
    return output

if __name__ == "__main__":
    videos = glob("videos/*")
    print(videos)
    for video in videos:
        output_path = get_output(video)
        cap = cv.VideoCapture(video)
        length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        mid = int(length/2)
        cap.set(cv.CAP_PROP_POS_FRAMES, mid)
        
        ret, frame = cap.read()

        if ret:
            cv.imwrite(output_path, frame)
        else:
            print(f"Error: Could not read frame {mid}")

        # Release resources
        cap.release()
        cv.destroyAllWindows()
