import numpy as np
import cv2
from glob import glob

def visualize_flow(flow):
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 1] = 255

    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = angle * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

flows = glob("of_stacks/*")
midframes = glob("mid_frames/*")
                          
for i,flow in enumerate(flows):
    # Load optical flow data from numpy file
    midframe = midframes[i]
    flow_data = np.load(flow)
    print(flow_data.shape)
    reshaped_array = flow_data.reshape((*flow_data.shape[:2], 4, 2))

    # Split the array along the third axis
    split_arrays = np.split(reshaped_array, 4, axis=2)

    # Reshape each split array back to its original shape (240, 320, 2)
    original_frames = [split_array.squeeze() for split_array in split_arrays]
    
    for i, flower in enumerate(original_frames):
        print("a")
        # Visualize flow
        flow_image = visualize_flow(flower)
        cv2.imshow(str(i), flow_image)

    image = cv2.imread(midframe)
    print(np.mean(flow_data))
    print(np.std(flow_data))

    # Display the result
    cv2.imshow(midframe, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
