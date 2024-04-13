stacks = r"of_stacks\_Art_of_the_Drink__Flaming_Zombie_pour_u_nm_np2_fr_med_1.npy"
fusion = r"fusion\_Art_of_the_Drink__Flaming_Zombie_pour_u_nm_np2_fr_med_1.npy"

import numpy as np

stacks = np.load(stacks)
fusion = np.load(fusion)

print(f"{stacks.shape = }")
print(f"{fusion.shape = }")