import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


folder_num = np.load("folder_num.npy")
parent_directory = 'Runs/Flock-' + str(folder_num[0])
folder = parent_directory + "/data"


def normalize(array):
    return array / np.sqrt(np.sum(array * array, axis=1))


# Models:
class Vicsek:
    def __init__(model, r_c2, scale=True, continuity=True):  # r_c2: cut off distance squared
        model.name = "Vicsek"
        model.r_c2 = r_c2
        model.continuity = continuity
        model.scale = scale
        model.parameters = ["rc_2 = "+str(model.r_c2)]

    def eta(model, y):
        return y < model.r_c2


class ConstantSpeedVicsek:
    def __init__(model, r_c2, continuity=True):  # r_c2: cut off distance squared
        model.name = "Constant-Speed Vicsek"
        model.r_c2 = r_c2
        model.continuity = continuity
        model.scale = False
        model.parameters = ["rc_2 = "+str(model.r_c2)]

    def eta(model, y):
        return y < model.r_c2


class Smale:
    def __init__(model, beta=0.3, K=1, sigma=1, continuity=True):  # r_c2: cut off distance squared
        model.name = "Smale"
        model.K = K
        model.sigma = sigma
        model.beta = beta
        model.continuity = continuity
        model.scale = False
        model.parameters = ["K = "+str(model.K), ", sigma = "+str(model.sigma), ", beta = " + str(model.beta)]

    def eta(model, y):
        return model.K/(model.sigma+y)**model.beta


# A flock of N birds with dynamics based on a model given as an argument
class Flock:
    def __init__(self, N=2, dimension=2, model=Vicsek(1), total_steps=1000, step_size=0.001,
                 initial_conditions="normal", Gamma=False, Lambda=False, measure_rate=1, animate=False):  # N = number of particles

        # Dynamics model: Vicsek, Constant-Speed_Vicsek and Smale
        self.model = model

        # Activate Measurement Devices
        self.measure_rate = measure_rate

        # Activate Animation
        self.animate = False

        if dimension == 2:
            self.animate = animate

        # System Properties
        self.N = N
        self.d = dimension

        # Initial Conditions
        if initial_conditions == "normal":
            self.x = np.random.normal(0, 1, (N, dimension))
            self.v = np.random.normal(0, 1, (N, dimension))

        else:
            self.x = initial_conditions[0]
            self.v = initial_conditions[1]

        if Gamma:
            self.gamma_stat(Gamma)

        if Lambda:
            self.lambda_stat(Lambda)

        if model == "Constant-Speed Vicsek":
            self.v = normalize(self.v)

        # self.com_frame()

        # Laplacian
        self.L = np.zeros((N, N))

        # Time
        self.continuous_t = 0
        self.discrete_t = 0
        self.h = step_size
        self.step = 0
        self.total_steps = total_steps

        # Observables
        self.E = 0  # energy
        self.P = 0  # momentum (d components)
        self.K = 0  # kinetic energy
        self.Gamma = 0  # sum of square of position differences of all pair of birds
        self.Lambda = 0  # sum of square of velocity differences of all pair of birds,
        # when Lambda is zero the flock has reached a uni-velocity state
        self.R_cm = 0  # center of mass (d components)
        self.V_cm = 0  # velocity of the center of mass (d components)
        self.observables = 5 + 3 * dimension  # 4+3*dimension observables plus one slot for time

        # storing data
        self.data = np.ndarray((self.total_steps, self.observables))

        # Animation Canvas
        self.fig, self.ax = plt.subplots(1, 1)
        self.line, = self.ax.plot([], [], 'b.', ms=5)

        # Start Command
        # self.Error = False  # Error turns True when an error comes up, we have
        # to restart the system from where it was left off
        self.start()

    def start(self):

        # if self.animate:
        #    ani = animation.FuncAnimation(self.fig, self.time_step,
        #                                  interval=25, blit=True)

        #    ani.save(folder+'/basic_animation.html', fps=30, extra_args=['-vcodec', 'libx264'])

        # else:
        while self.step < self.total_steps:
            self.time_step()
            self.step += 1

        np.save(folder + "/measurements.npy", self.data)

        info_file = open(parent_directory+"/info.txt", "w")
        info_file.writelines(["N="+str(int(self.N)), "\ndimension="+str(int(self.d)), "\nmodel: "+self.model.name, "\nparameters: "])
        info_file.writelines(self.model.parameters)
    #  def restart(self):

    #    if self.Error:
    #        data = np.load(folder + '/state.npy')
    #        # feeding in the last saved positions and velocities
    #        self.x = data[0:self.N]
    #        self.v = data[self.N:]

    #    self.start()

    # def com_frame(self):  # transform velocities to center-of-mass frame in order to maintain momentum conservation
    #    self.v -= self.V_cm

    # def velocity_autocorrelation(self):

    def Laplacian(self, x_array):  # calculating the accelerations
        self.L = np.zeros((self.N, self.N))
        self.Gamma = 0
        # first method:
        for i in range(self.N):
            for j in range(i + 1, self.N):
                x_ij = x_array[i] - x_array[j]
                y = np.dot(x_ij, x_ij)
                self.Gamma += y
                a_ij = self.model.eta(y)
                self.L[i, j] = a_ij
                self.L[j, i] = a_ij
                self.L[i, i] -= a_ij
                self.L[j, j] -= a_ij

        if self.model.scale:
            self.L = self.L/(-np.diag(self.L)+1)
        # second method:
        """
        for i in range(self.N):
            for j in range(i + 1, self.N):
                x_ij = self.x[i] - self.x[j]
                y = x_ij * x_ij
                a_ij = self.model.eta(y)         
        """

    def time_step(self):  # moving one step forward in time

        # calculating initial accelerations
        self.Laplacian(self.x)

        if self.model.continuity:
            # continuous: 4th order Runge-Kutta
            v1 = self.v
            k1 = np.dot(self.L, v1)

            self.Laplacian(self.x + self.h * self.v/2)
            v2 = self.v + self.h * k1 / 2
            k2 = np.dot(self.L, v2)

            self.Laplacian(self.x + self.h * v2/2)
            v3 = self.v + self.h * k2 / 2
            k3 = np.dot(self.L, v3)

            self.Laplacian(self.x + self.h * v3)
            v4 = self.v + self.h * k3 / 2
            k4 = np.dot(self.L, v4)

            self.x = self.x + self.h*(v1+2*v2+2*v3+v4)/6
            self.v = self.v + self.h*(k1+2*k2+2*k3+k4)/6
            self.continuous_t += self.h

        else:
            # discrete:
            self.x = self.x + self.v
            self.v = self.v + np.dot(self.L, self.v)
            self.discrete_t += 1

        # readjusting velocities to conserve momentum
        # self.com_frame()

        if self.step % self.measure_rate == 0:

            # measurements
            self.E = -np.sum(self.v * np.dot(self.L, self.v))
            self.K = np.sum(self.v * self.v)
            self.P = np.sum(self.v, axis=0)
            self.Lambda = self.N * self.K - np.dot(self.P, self.P)
            self.R_cm = np.mean(self.x, axis=0)
            self.V_cm = self.P/self.N
            # self.Gamma = measurement done in Laplacian

            # storing data
            k = self.step//self.measure_rate
            self.data[k] = [self.continuous_t, self.Gamma, self.Lambda, self.K, self.E, *self.P, *self.R_cm, *self.V_cm]
            print(self.data[k])

        # saving phase
        # data = np.concatenate((self.x, self.v))
        # np.save(folder + "/phase.npy", data)

        # animate
        # if self.animate:
        #    return self.line.set_data(self.x[:, 0], self.x[:, 1])

    def lambda_stat(self, Lambda):
        self.v = self.v * np.sqrt(Lambda/self.Lambda)

    def gamma_stat(self, Gamma):
        self.x = self.x * np.sqrt(Gamma/self.Gamma)

    def equilibrium(self, observable):
        self.data[k]