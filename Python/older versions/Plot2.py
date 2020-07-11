import numpy as np
import os
import scipy.optimize as opt
import matplotlib.pyplot as plt

folder_num = np.load("folder_num.npy")[0]
folder = 'Flock-'+str(folder_num)
data = np.load(folder+"/measurements.npy")
fig, ax = plt.subplot()


#self.data.append(np.array(
#    [[self.M_bar, self.M_error],
#     [self.X, self.X_error],
#     [self.E_bar, self.E_error],
#     [self.C_V, self.C_V_error],
#     [self.Xi, self.Xi_error],
#     [self.beta, self.tau]]))

merged = True


def exp_decay(x, tau):
    return np.exp(-x / tau)


def autocorrelation(x, temp, s1='M', s2='t'):
    s = np.alen(x)
    xi = np.ndarray(int(4*s/5))
    ### Runtime error
    for j in range(int(4*s/5)):
        xi[j] = np.mean(x[:s-j]*x[j:]) - np.mean(x[:s-j])*np.mean(x[j:])

    xi /= np.var(x)
    ydata = xi
    ydata = ydata[np.logical_not(np.isnan(ydata))]
    ydata = ydata[np.logical_not(np.isinf(ydata))]
    xdata = np.arange(np.alen(ydata))+1
    popt, pcov = opt.curve_fit(exp_decay, xdata, ydata)
    fig1, ax = plt.subplots(1, 1)
    fig1.set_size_inches((4, 3))
    ax.plot(xdata, exp_decay(xdata, *popt), 'r-', label='fit: \u03c4=%.3g' % popt[0])
    ax.scatter(xdata, ydata, s=2)
    ax.set_title('Autocorrelation of ' + s1)
    ax.set_xlabel(s2)
    ax.set_ylabel('a_c')
    ax.text(0.0, 1.0, '\u03b2 = %.2f' % temp, horizontalalignment='left', verticalalignment='bottom',
            transform=ax.transAxes)
    ax.legend(loc=1, fontsize=8)
    plt.tight_layout()
    fig1.savefig(folder+'/%.2f'%temp+'-autocorrelation.png')
    plt.close(fig1)
    return int(popt[0]+1)


def correlation(x, temp, draw = False):
    l = np.alen(x[:,0])
    xi = np.ndarray(l-1)

    for j in range(l-1):
        xi[j] = np.mean(x[0] * x[j+1]) - np.mean(x[0]) * np.mean(x[j+1])

    ydata = xi[np.where(xi!=0)]
    ydata = ydata[np.logical_not(np.isnan(ydata))]
    ydata = ydata[np.logical_not(np.isinf(ydata))]
    ydata = ydata[np.logical_not(ydata==0)]
    xdata = np.arange(np.alen(ydata))+1
    logx = np.log10(xdata)
    logy = np.log10(ydata)
    slope, intercept, r, p, stderr = sts.linregress(logx,logy)
    if draw:
        fig2, ax2 = plt.subplots(1, 1)
        fig2.set_size_inches((4, 3))
        ax2.plot(logx, slope* logx + intercept, 'r-', label='fit \u03be = %.2f'%slope)
        ax2.scatter(logx, logy, s=2)
        ax2.set_title('Spatial Correlation')
        ax2.set_xlabel('(horizontal) distance')
        ax2.set_ylabel('correlation')
        ax2.text(0.0, 1.0, '\u03b2 = %.2f' % temp, horizontalalignment='left', verticalalignment='bottom',
                transform=ax2.transAxes)
        ax2.legend(loc=1, fontsize=8)
        plt.tight_layout()
        fig2.savefig(folder+'/%.2f'%temp+'-spatial-correlation.png')
        plt.close(fig2)
    return slope


def plot(xdata, ydata, error_bar=[], title='', x_label='', y_label=''):
    if merged:
        data = np.array([[xdata[i], ydata[i]] for i in np.argsort(xdata)])
        xdata = data[:, 0]
        ydata = data[:, 1]

    x_max = xdata[np.argmax(ydata)]
    y_max = np.max(ydata)
    np.save(folder+"/max"+title+".npy",np.array([x_max, y_max]))
    fig3 = plt.figure(figsize=(2.8,2.1))
    ax3 = fig3.add_subplot(111)
    if error_bar!=[]:
        ax3.errorbar(xdata, ydata, yerr=error_bar, ecolor='red', linewidth=1)
        err = '_error'
    else:
        ax3.scatter(xdata, ydata,s=2)
        err = ''

    ax3.set_title(title, fontsize=10)
    ax3.set_xlabel(x_label, fontsize=8)
    ax3.set_ylabel(y_label, fontsize=8)
    ax3.tick_params(axis='both', which='major', labelsize=8)
    plt.tight_layout()
    fig3.savefig(folder+'/'+title+err+'.png')
    plt.close(fig3)


ydata = np.load(folder+file)
beta = ydata[:,5,0]
M_data = abs(ydata[:,0,0])
M_error = ydata[:,0,1]
X_data = ydata[:,1,0]
X_error = ydata[:,1,1]
E_data = ydata[:,2,0]
E_error = ydata[:,2,1]
C_data = ydata[:,3,0]
C_error = ydata[:,3,1]
Xi_data = ydata[:,4,0]
Xi_error = ydata[:,4,1]

d = np.size(data, axis=1)
abbreviations = ["G", "L", "K", "E", "P", "C", "V"]
abb_to_num = dict({"G":0, "L":1, "K":2, "E":3, "P":4, "C":4+d, "V":4+2*d})
name = ["Gamma", "Lambda", "Kinetic Energy", "Energy", "Momentum", "CoM", "V_CoM"]  # 7 observables
dim = [1, 1, 1, 1, d, d, d]
class Observable:
    def __init__(self, abb):
        self.abbreviation = abb
        self.num = abb_to_num[abb]
        self.dim = dim[self.num]
        self.name = name[self.num]
        self.data = data[:,self.num:self.num+self.dim]

observable_list=[]
for abb in abbreviations:
    observable_list.append(Observable(abb))

for quantity in observables:

plot(beta,M_data, error_bar=M_error, title='Magnetization', x_label='\u03b2', y_label='|M|')
plot(beta,M_data, title='Magnetization', x_label='\u03b2', y_label='|M|')
plot(beta,X_data, error_bar=X_error, title='Susceptibility', x_label='\u03b2', y_label='\u03c7')
plot(beta,X_data, title='Susceptibility', x_label='\u03b2', y_label='\u03c7')
plot(beta,E_data, error_bar=E_error, title='Energy', x_label='\u03b2', y_label='E')
plot(beta,E_data, title='Energy', x_label='\u03b2', y_label='E')
plot(beta,C_data, error_bar=C_error, title='Heat Capacity', x_label='\u03b2', y_label='C')
plot(beta,C_data, title='Heat Capacity', x_label='\u03b2', y_label='C')
plot(beta,Xi_data, error_bar=Xi_error, title='Correlation Length', x_label='\u03b2', y_label='\u03be')
plot(beta,Xi_data, title='Correlation Length', x_label='\u03b2', y_label='\u03be')

#M = np.load(folder+"/M.npy")
#m = M[:,1]
#l = np.alen(m)
#time = M[:,0]
#c = np.ndarray((l))
#for i in range(l):
#    c[i] = 9534

#fig4 = plt.figure(figsize=(3.6,2.7))
#ax4 = fig4.add_subplot(111)
#ax4.plot(time,m,'-b')
#ax4.plot(time,c,'-r',label="mean = 9534")
#ax4.set_xlabel("t [MC steps]")
#ax4.set_ylabel("M")
#ax4.legend()
#ax4.set_title("magnetization")
#plt.tight_layout()
#fig4.savefig(folder+"m.png")