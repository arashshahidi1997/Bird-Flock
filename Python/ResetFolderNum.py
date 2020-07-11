import numpy as np
folder_num = [1]
np.save("folder_num.npy", np.array(folder_num, dtype='int'), allow_pickle=True)

