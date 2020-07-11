import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math

dots = np.random.rand(2500,4)
dots[:,2:] -= 0.5
dots[:,2:] *= 0.01


def com_frame(v_x, v_y):  # transform velocities to center of mass frame in order to maintain momentum conservation
    v_x -= v_cm_x
    v_y -= v_cm_y
    return v_x, v_y


def v_com(v_x, v_y):  # calculate the center of mass velocity
    return np.mean(v_x), np.mean(v_y)

h = 0.0001 # time steps
h2 = h*h # h squared. I try to calculate all recurring terms to reduce the number of operations in the main loop
v_max = 10 # maximum velocity
r_c = 2.5 # cutoff length
d = 2 # dimension of the sytem
N = 100 # number of particles
total_steps = 100 # number of time steps that we let the sytem run through

# sytem length and volume
L = 3 * np.sqrt(N)
V = L * L

# initial positions
N = N
x = np.linspace(0, L / 2, N)
y = np.linspace(0, L, N)

# initial velocities
v_max = v_max
v_x = v_max * (2 * np.random.rand(N) - 1)
v_y = v_max * (2 * np.random.rand(N) - 1)
v_cm_x, v_cm_y = v_com(v_x, v_y)
v_x, v_y = com_frame(v_x, v_y)

# accelerations
a_x = np.zeros(N)
a_y = np.zeros(N)

# time
t = 0

# observables
E = 0  # energy
U = 0  # potential energy (Leonard-Jones)
K = 0  # kinetic energy
T = 0  # temperature
P = 0  # pressure
P_2nd_virial = 0  # 2nd term of pressure coming from Virial theorem
N = N  # number of particles


def acceleration():  # calculating the accelerations
    a_x = np.zeros(N)
    a_y = np.zeros(N)
    P_2nd_virial = 0

    for i in range(N):
        for j in range(i + 1, N):
            # shortest distance between the class of images of two particles. each particle has a class of images due to periodic boundary conditions
            x_ij = math.remainder(x[i] - x[j], L)
            if x_ij < r_c:  # r_c = cutoff length
                y_ij = math.remainder(y[i] - y[j], L)
                if y_ij < r_c:
                    # the i_th and j_th particle are close enough to interact
                    r_ij = x_ij * x_ij + y_ij * y_ij
                    r_ij2 = r_ij * r_ij
                    r_ij6 = r_ij2 * r_ij2 * r_ij2
                    r_ij12 = r_ij6 * r_ij6
                    alpha = -2 * (-12 / r_ij12 + 6 / r_ij6) / r_ij2
                    U += 4 * (1 / r_ij12 - 1 / r_ij6)
                    a_x = alpha * x_ij
                    a_y = alpha * y_ij
                    P_2nd_virial += alpha * r_ij2
                    a_x[i] += a_x
                    a_x[j] -= a_x  # Newton's 3rd law
                    a_y[i] += a_y
                    a_y[j] -= a_y

    return a_x, a_y

def main():
    fig, ax = plt.subplots()

    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    
    x = dots[:,0]
    y = dots[:,1]
    
    line, = ax.plot([],[],'b.',ms=5)
    
    def animate(i):
        #velocity verlet part 1

        #velocity Verlet part 1
        for i in range(N):
            x[i] = (x[i] + v_x[i] * h + a_x[i] * h2) % L
            y[i] = (y[i] + v_y[i] * h + a_y[i] * h2) % L
            v_x[i] += a_x[i] * h
            v_y[i] += a_y[i] * h

        #force calculation
        acceleration()

        #velocity verlet part 2
        for i in range(N):
            v_x[i] = a_x[i] * h
            v_y[i] = a_y[i] * h

        #thermostat
        #measurements
        #save data
        
        line.set_data(x,y)  # update the data
        return line,
    
    ani = animation.FuncAnimation(fig, animate, 
                                  interval=25, 
                                  blit=True)

    ani.save('basic_animation.html', fps=30, extra_args=['-vcodec', 'libx264'])
    
    plt.show()

def timestep():
    dots[:,:2] += dots[:,2:]

    # gravity
    # dots[:,3]  -= 0.001

    #collision()

    cross_left  = dots[:,0] < 0.
    cross_right = dots[:,0] > 1.
    cross_top   = dots[:,1] > 1.
    cross_bot   = dots[:,1] < 0.

    dots[cross_left | cross_right, 2] *= -1.
    dots[cross_top  | cross_bot,   3] *= -1.

    dots[cross_left, 0] = 0.
    dots[cross_right,0] = 1.

    dots[cross_top, 1] = 1.
    dots[cross_bot, 1] = 0.


def collision():
    from scipy.spatial.distance import pdist, squareform
    D = squareform(pdist(dots[:, :2]))
    ind1, ind2 = np.where(D < 0.01)

    unique = (ind1 < ind2)
    ind1 = ind1[unique]
    ind2 = ind2[unique]

    for i1, i2 in zip(ind1, ind2):
        
        # location vector
        r1 = dots[i1, :2]
        r2 = dots[i2, :2]
        # velocity vector
        v1 = dots[i1, 2:]
        v2 = dots[i2, 2:]
        # relative location & velocity vectors
        r_rel = r1 - r2
        v_rel = v1 - v2
        # vector of the center of mass
        v_cm = (v1 + v2) / 2.
        # collisions of spheres reflect v_rel over r_rel
        rr_rel = np.dot(r_rel, r_rel)
        vr_rel = np.dot(v_rel, r_rel)
        v_rel = v_rel - 2 * r_rel * vr_rel / rr_rel
        # assign new velocities
        dots[i1, 2:] = v_cm + 0.5 * v_rel
        dots[i2, 2:] = v_cm - 0.5 * v_rel
        # crude avoidance of glue effect
        dots[i1, :2] += 0.1*r_rel
        dots[i2, :2] -= 0.1*r_rel

main()


# Based on 
#
# Animation of Elastic collisions with Gravity
# 
# author: Jake Vanderplas
# email: vanderplas@astro.washington.edu
# website: http://jakevdp.github.com
# license: BSD
# Please feel free to use and modify this, but keep the above information. Thanks!


