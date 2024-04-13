
import numpy as np
import matplotlib.pyplot as plt

def get_of(mid_frame :str):
    return mid_frame.split(".")[0]+"_opticalflow_0.npy"


from glob import glob
import numpy as np

def normalize_image_per_channel(image, mean_per_channel, std_per_channel):
    """
    Normalize the channels of the image array to have given mean and standard deviation per channel.

    Args:
    - image: NumPy array representing an image with shape (height, width, channels)
    - mean_per_channel: List or array containing mean values for each channel
    - std_per_channel: List or array containing standard deviation values for each channel

    Returns:
    - Normalized image array
    """
    # Ensure image is in float32 format for accurate calculations
    image = image.astype(np.float32)

    # Iterate over each channel and normalize
    for channel in range(image.shape[2]):
        channel_mean = mean_per_channel[channel]
        channel_std = std_per_channel[channel]
        # Normalize the channel
        image[:, :, channel] = ((image[:, :, channel] - np.mean(image[:, :, channel])) / np.std(image[:, :, channel])) * channel_std + channel_mean

    return image

for annotation in ["mid_frame_test.csv", "mid_frame_train.csv", "mid_frame_val.csv"]:
    print(annotation)
    with open(annotation) as csvfile:
        for row in csvfile:
            filename, label = row.split(",")
            of_filename = get_of(filename)
            of_file = np.load("optical_flow_mid_frames/" + of_filename)
            image = plt.imread('mid_frames/' + filename)  
            image_array = np.array(image)
            image_array = normalize_image_per_channel(image_array, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            stacked_array = np.dstack((image_array,of_file))
            np.save("fusion/" + filename, stacked_array)

fusions = glob("fusion/*")
for fusion in fusions:
    print(np.load(fusion).shape)