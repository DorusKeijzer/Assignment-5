from glob import glob
from numpy import load, max, mean, std, save

ofs = glob("optical_flow/*")

for of in ofs:
    array = load(of)
    mean_value = mean(array)
    std_dev = std(array)

    # Normalize the arrayay
    normalized_array = (array - mean_value) / std_dev
    save(of, normalized_array)
