import torch
import cv2
from dataset import optical_flow_test_dataloader as dataloader
import numpy as np
def visualize_flow(flow):
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 1] = 255

    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = angle * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def visualize_optical_flow_batch(batch):
    images, labels = batch
    num_samples = len(images)
    
    for i in range(num_samples):
        flow_image = images[i].numpy()
        print(flow_image)
        print(flow_image.shape)
        label = labels[i].item()
        
        # Convert flow image to BGR for visualization
        flow_image_bgr = visualize_flow(flow_image)
        
        # Display the flow image and its label
        cv2.imshow(f"{label}", flow_image_bgr)
    
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Assuming you have a DataLoader named 'dataloader'
for batch in dataloader:
    visualize_optical_flow_batch(batch)
