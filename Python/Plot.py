import numpy as np
import os
import scipy.optimize as opt
import matplotlib.pyplot as plt
import scipy.stats as sts


def figures(folder_num, tol=10**-3):
    folder = 'Runs/Flock-'+str(folder_num)
    data = np.load(folder+"/data/measurements.npy")
    folder += '/figures'
    os.mkdir(folder)

    class Observable:
        def __init__(self, abb):
            self.abbreviation = abb
            self.num, self.dim = abb_to_num[abb]
            self.name = name[abbreviations.index(abb)]

            self.data = data[:, self.num:self.num + self.dim]

            self.equilibrium_array = self.data[1:]/self.data[:-1]-1 < tol
            self.equilibrium = np.sum(self.equilibrium_array[:-5], axis=0) == 5

            if self.dim == 1:
                if self.equilibrium:
                    self.fixed_value = self.data[-1]
                else:
                    self.fixed_value = np.NaN

            else:
                self.fixed_value = np.ndarray(self.dim)

                for component in range(self.dim):
                    if self.equilibrium[component]:
                        self.fixed_value[component] = self.data[-1, component]

                    else:
                        self.fixed_value[component] = np.NaN

    def plot(xdata, ydata, error_bar=[], title='', x_label='', y_label=''):

        # if merged:
        #   data = np.array([[xdata[i], ydata[i]] for i in np.argsort(xdata)])
        #    xdata = data[:, 0]
        #    ydata = data[:, 1]

        # x_max = xdata[np.argmax(ydata)]
        # y_max = np.max(ydata)
        # np.save(folder+"/max"+title+".npy",np.array([x_max, y_max]))

        fig3 = plt.figure(figsize=(2.8, 2.1))
        ax3 = fig3.add_subplot(111)
        if error_bar!=[]:
            ax3.errorbar(xdata, ydata, yerr=error_bar, ecolor='red', linewidth=1)
            err = '_error'
        else:
            ax3.scatter(xdata, ydata, s=2)
            err = ''

        ax3.set_title(title, fontsize=10)
        ax3.set_xlabel(x_label, fontsize=8)
        ax3.set_ylabel(y_label, fontsize=8)
        ax3.tick_params(axis='both', which='major', labelsize=8)
        plt.tight_layout()
        fig3.savefig(folder+"/"+title+err+'.png')
        plt.close(fig3)

    d = (np.size(data, axis=1)-5)//3
    abbreviations = ["G", "L", "K", "E", "P", "C", "V"]
    abb_to_num = dict({"G": [1, 1], "L": [2, 1], "K": [3, 1], "E": [4, 1], "P": [5, d], "C": [5+d, d], "V": [5+2*d, d]})
    name = ["Gamma", "Lambda", "Kinetic Energy", "Energy", "Momentum", "CoM", "V_CoM"]  # 7 observables

    observable_list = []
    fixed_value_list = []

    for abb in abbreviations:
        observable_list.append(Observable(abb))
        fixed_value_list.append(observable_list[-1].fixed_value)

    for quantity in observable_list:
        if quantity.dim == 1:
            plot(data[:, 0], quantity.data, title=quantity.name, x_label='t', y_label=quantity.abbreviation)

        else:
            for i in range(quantity.dim):
                plot(data[:, 0], quantity.data[:, i], title=quantity.name + '-' + str(i), x_label='t',
                     y_label=quantity.abbreviation + '-' + str(i))

    print(fixed_value_list)

    def exp_decay(x, tau):
        return np.exp(-x / tau)

    def autocorrelation(x, temp, s1='M', s2='t'):

        s = np.alen(x)
        xi = np.ndarray(int(4*s/5))

        # Runtime error
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

    def correlation(x, temp, draw=False):
        l = np.alen(x[:, 0])
        xi = np.ndarray(l-1)

        for j in range(l-1):
            xi[j] = np.mean(x[0] * x[j+1]) - np.mean(x[0]) * np.mean(x[j+1])

        ydata = xi[np.where(xi != 0)]
        ydata = ydata[np.logical_not(np.isnan(ydata))]
        ydata = ydata[np.logical_not(np.isinf(ydata))]
        ydata = ydata[np.logical_not(ydata == 0)]
        xdata = np.arange(np.alen(ydata))+1
        logx = np.log10(xdata)
        logy = np.log10(ydata)
        slope, intercept, r, p, stderr = sts.linregress(logx, logy)
        if draw:
            fig2, ax2 = plt.subplots(1, 1)
            fig2.set_size_inches((4, 3))
            ax2.plot(logx, slope * logx + intercept, 'r-', label='fit \u03be = %.2f'%slope)
            ax2.scatter(logx, logy, s=2)
            ax2.set_title('Spatial Correlation')
            ax2.set_xlabel('(horizontal) distance')
            ax2.set_ylabel('correlation')
            ax2.text(0.0, 1.0, '\u03b2 = %.2f' % temp, horizontalalignment='left', verticalalignment='bottom',
                     transform=ax2.transAxes)
            ax2.legend(loc=1, fontsize=8)
            plt.tight_layout()
            fig2.savefig(folder + '/%.2f' % temp + '-spatial-correlation.png')
            plt.close(fig2)
        return slope

