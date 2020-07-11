import numpy as np
import pandas as pd
import math
import os
import matplotlib.pyplot as plt
from matplotlib import animation
import scipy.optimize as opt

# Argon, velocity-Verlet, NVE ensemble, LJ potential, MD simulation

folder = 'MD-11'
try:
    os.mkdir(folder)
except:
    exception = 0

h = 0.0001 # time steps
h2 = h*h # h squared. I try to calculate all recurring terms to reduce the number of operations in the main loop
v_max = 10 # maximum velocity
r_c = 2.5 # cutoff length
d = 2 # dimension of the system
N = 100 # number of particles
total_steps = 100 # number of time steps that we let the system run through

Flag = False # Flag is turned True if an error comes up so we have to restart the system from where it was left off

def exp_decay(x, tau):
    return np.exp(-x / tau)

def autocorrelation(x, text, s1, s2='t'):
    s = int(np.alen(x)/2)
    ac = np.ndarray(s)
    ### Runtime error
    for j in range(s):
        ac[j] = np.mean(x[:s-j]*x[j:s]) - np.mean(x[:s-j])*np.mean(x[j:s])

    ac /= np.var(x)
    ydata = ac
    ydata = ydata[np.logical_not(np.isnan(ydata))]
    ydata = ydata[np.logical_not(np.isinf(ydata))]
    xdata = np.arange(np.alen(ydata))+1
    popt, pcov = opt.curve_fit(exp_decay, xdata, ydata)
    fig_ac = plt.figure(figsize=(2.8,2.1))
    ax_ac = fig_ac.add_subplot(111)
    fig_ac.set_size_inches((4, 3))
    ax_ac.plot(xdata, exp_decay(xdata, *popt), 'r-', label='fit: \u03c4=%.3g' % popt[0])
    ax_ac.scatter(xdata, ydata, s=2)
    ax_ac.set_title('Autocorrelation of ' + s1)
    ax_ac.set_xlabel(s2)
    ax_ac.set_ylabel('a_c')
    ax_ac.text(0.0, 1.0, '\u03b2 = %.2f' % text, horizontalalignment='left', verticalalignment='bottom',
            transform=ax_ac.transAxes)
    ax_ac.legend(loc=1, fontsize=8)
    plt.tight_layout()
    fig_ac.savefig(folder+'/%.2f'% text +'-autocorrelation.png')
    plt.close(fig_ac)
    return popt[0]


class NVE_LJ_Argon: # class of N Argon atoms in a d dimensional box of constant volume, interacting with a Leonard-Jones potential, supposedly conserving the total energy of the system

    def __init__(self, N, measure=False, animate=False): # N = number of particles
        # activate measurement devices
        self.measure = measure
        self.equilibrium = False

        # activate animation
        self.animate = animate

        # system length and volume
        self.L = 3 * np.sqrt(N)
        self.V = self.L * self.L

        # initial positions
        self.N = N
        self.x = np.linspace(0, self.L/2, N)
        self.y = np.linspace(0, self.L, N)

        # initial velocities
        self.v_max = v_max
        self.v_x = self.v_max * (2*np.random.rand(N)-1)
        self.v_y = self.v_max * (2*np.random.rand(N)-1)
        self.v_list = []
        self.ac_v = []
        self.v_cm_x, self.v_cm_y = self.v_com()
        self.com_frame()

        # accelerations
        self.a_x = np.zeros(N)
        self.a_y = np.zeros(N)

        # time
        self.t = 0
        self.delta_t = 0

        # observables
        self.E = 0 # energy
        self.U = 0 # potential energy (Leonard-Jones)
        self.K = 0 # kinetic energy
        self.T = 0 # temperature
        self.P = 0 # pressure
        self.P_2nd_virial = 0 # 2nd term of pressure coming from Virial theorem
        self.N= N # number of particles

        # animation canvas
        self.fig, self.ax = plt.subplots(1,1)
        self.line, = self.ax.plot([],[],'b.',ms=5)

        # start command
        self.start()


    def start(self):

        if self.animate:
            ani = animation.FuncAnimation(self.fig, self.time_step,
                                                 interval=25, blit=True)

            ani.save('basic_animation.html', fps=30, extra_args=['-vcodec', 'libx264'])

        else:
            for i in range(total_steps):
                self.time_step()

    def restart(self):

        if Flag:
            data = np.load(folder+'/phase.npy')
            # feeding in the last saved positions and velocities
            self.x = data[:,0]
            self.y = data[:,1]
            self.v_x = data[:,2]
            self.v_y = data[:,3]

        self.start()

    def v_com(self): # calculate the center of mass velocity
        return np.mean(self.v_x), np.mean(self.v_y)

    def com_frame(self): # transform velocities to center of mass frame in order to maintain momentum conservation
        self.v_x -= self.v_cm_x
        self.v_y -= self.v_cm_y

    def velocity_autocorrelation(self):
        self.v
        
    def acceleration(self): # calculating the accelerations
        self.a_x = np.zeros(N)
        self.a_y = np.zeros(N)
        self.P_2nd_virial = 0
        self.U = 0

        for i in range(self.N):
            for j in range(i+1,self.N):
                # shortest distance between the class of images of two particles. each particle has a class of images due to periodic boundary conditions
                x_ij = math.remainder(self.x[i]-self.x[j], self.L)
                if x_ij < r_c: # r_c = cutoff length
                    y_ij = math.remainder(self.y[i]-self.y[j], self.L)
                    if y_ij < r_c:
                        # the i_th and j_th particle are close enough to interact
                        r_ij = x_ij*x_ij + y_ij*y_ij
                        r_ij2 = r_ij * r_ij
                        r_ij6 = r_ij2 * r_ij2 * r_ij2
                        r_ij12 = r_ij6 * r_ij6
                        alpha = -2 * (-12/r_ij12 + 6/r_ij6)/r_ij2
                        self.U += 4 * (1/r_ij12 - 1/r_ij6)
                        a_x = alpha * x_ij
                        a_y = alpha * y_ij
                        self.P_2nd_virial += alpha * r_ij2
                        self.a_x[i] += a_x
                        self.a_x[j] -= a_x # Newton's 3rd law
                        self.a_y[i] += a_y
                        self.a_y[j] -= a_y

    def time_step(self): # moving one step forward in time

        # calculating initial accelerations
        self.acceleration()

        # velocity Verlet part 1
        for i in range(self.N):
            self.x[i] = (self.x[i] + self.v_x[i] * h + self.a_x[i] * h2) % self.L
            self.y[i] = (self.y[i] + self.v_y[i] * h + self.a_y[i] * h2) % self.L
            self.v_x[i] += self.a_x[i] * h
            self.v_y[i] += self.a_y[i] * h

        # force calculation, also calculating instantaneous potential energy and 2nd term of pressure.
        self.acceleration()

        #velocity Verlet part 2
        for i in range(self.N):
            self.v_x[i] = self.a_x[i] * h
            self.v_y[i] = self.a_y[i] * h

        # readjusting velocities to conserve momentum
        self.com_frame()

        # measurements
        self.K = (np.dot(self.v_x, self.v_x) + np.dot(self.v_y, self.v_y))/2
        self.T = self.K/(self.N-1)
        self.P = (self.N * self.T - self.P_2nd_virial/d)/self.V
        self.data.append([self.K, self.P, self.U])

        # integrating over time
        if self.measure and self.equilibrium:
            U += self.U*h # integrating instantaneous potential energy w.r.t time
            K += self.K*h # integrating instantaneous potential energy w.r.t time
            P += self.P*h #
            self.delta_t += h
        self.t += h

        # save
        data = np.array([self.x, self.y, self.v_x, self.v_y])
        np.save(folder+"phase.npy", data)
        if self.measure:
             np.save(folder + "/measurements.npy", np.array([self.t, K/self.delta_t, P/self.delta_t, U/self.delta_t]))

        # animate
        if self.animate:
            return self.line.set_data(self.x,self.y)
try:
    Box = NVE_LJ_Argon(10, measure=True, animate=False)
except:
    Flag = True
    Box.restart()