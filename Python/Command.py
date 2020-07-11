from Flocks import *
from Plot import *
from Ensemble import *

folder_num = np.load("folder_num.npy")
parent_directory = 'Runs/Flock-' + str(folder_num[0])
os.mkdir(parent_directory)
folder = parent_directory + "/data"
os.mkdir(folder)

flock = Flock(N=100, total_steps=1000, step_size=0.001, dimension=2, model=Smale(beta=0.3, K=1, sigma=1))

# observable-vs.-time plots
figures(folder_num[0])

# smale ensemble
# if flock.model.name == "Smale":
#    smale_ensemble(flock)


folder_num[0] = folder_num[0] + 1
np.save("folder_num.npy", folder_num)

