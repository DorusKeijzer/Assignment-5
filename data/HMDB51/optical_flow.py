import cv2
from glob import glob
import numpy as np

def get_output(video_path:str)-> str:
    filename = video_path.split("\\")[-1]
    output = "optical_flow" + filename[:-4] + "_opticalflow.avi"
    return output

def calculate_optical_flow(video_path: str, output_path: str) -> None:
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Initialize variables for the first frame
    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame1.shape[1], frame1.shape[0]))

    while(cap.isOpened()):
        # Read the current frame and convert it to grayscale
        ret, frame2 = cap.read()
        if not ret:
            break
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Calculate the optical flow using Lucas-Kanade method
        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Convert optical flow to polar coordinates for visualization
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        # Convert HSV to BGR for visualization
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Write the frame with optical flow visualization to the output video
        out.write(bgr)

        # Display the frame
        cv2.imshow('Optical flow', bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Update the previous frame
        prvs = next

    # Release the video capture and writer objects
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    files = glob("videos/*")
    for file in files:
        print(file)
        output = get_output
        calculate_optical_flow(file, output)