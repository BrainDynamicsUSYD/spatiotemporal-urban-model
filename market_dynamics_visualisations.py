import numpy as np

import time
from datetime import datetime

from matplotlib import cm, pyplot as plt
import matplotlib as mpl
mpl.rc('font',family='Times New Roman',size= 14)
import matplotlib.markers as mmarkers
from matplotlib.colors import ListedColormap
import matplotlib.gridspec as gridspec
import seaborn as sns
import scipy
from scipy.ndimage.filters import gaussian_filter

from os.path import dirname, abspath
import os
import sys
import shutil
global dirmain, diroutput     # relative directories
dirmain = dirname(abspath(__file__))
diroutput = dirmain + '/visualisations/output/'
dirinput = dirmain + '/data/input/'

class household:
    def __init__(self, N, N_min):
        """ 
        __init__ initialises parameters and dictionaries for which characteristics and preferences are sampled from later.
        
        param N: input number of households 
        param N_min: absolute minimum number of households 

        """

        self.N = int(N)  
        self.N_min = int(N)       
        self.q_bound = {'n_people': np.linspace(1,4,4), 'type I/II': [1]}   # bounds for characteristics, 
                                            # type I/II is for testing different characteristics of households
        self.s_bound = {'s_0': [-1,0], 'classical:modern': [-1, 1], 'large house': [0,1]}   # bounds for preferences


class dwelling:
    def __init__(self, M, R, u0_max, u0_wid, price_scale, M_max):
        """ 
        __init__ initialises parameters and dictionaries for which characteristics are sampled from later.
        
        param M: input number of dwellings 
        param R: linear size of model
        param u0_max: maximum U0 value 
        param u0_wid: window size for U0 distribution
        param price_scale: self explanatory by name, value to scale model values to; default unity scale
        param M_max: max number of dwellings permitted

        """

        self.M = int(M)    
        self.R = R       
        self.r_coord = np.zeros((1, 2))     # blank coordinate array
        self.u0_max = float(u0_max)         
        self.u0_wid = u0_wid
        self.price_scale = price_scale   
        self.M_max = M_max    

        """ establish some dictionaries for later (will search by index of element to find same term to compare 
        utility of specific characteristics of dwelling) """
        
        self.style = {'classical:modern': [-1,1]}


class simulate:
    def __init__(self):
        self.all_store_pop = []
        self.all_store_dwell = []

        simulate.model_parameters(self) # instantiate the model parameters for simulation

    def model_parameters(self):
        """ 
        model_parameters() instantiates the main parameters in the model.

        """   

        # General parameters for model
        self.R = 25                         # size of coordinate space 
        self.t = 25                         # no. years for simulation
        self.t_inc = 4                      # increments per year; annualized rates will be divided by this 
        self.chi = 1                        # probability of transaction clearance (decimal)
        self.sigma = 0.06/self.t_inc        # fractional move rate (evaluated t_inc/year)

        # Parameters for population and dwelling increase/decrease (evaluated 1/year)
        # rates are as per evaluation period i.e. per year/no. steps per year
        self.eps = 0.00/self.t_inc         # rate of population growth 
        self.rho = 0.00/self.t_inc         # rate of population decline
        self.beta = 0.00/self.t_inc        # rate of dwelling creation
        self.lamb = 0.02/self.t_inc        # rate of dwelling destruction

        # affinity function G
        self.G_0 = 0.2
        self.h = 1

        # household parameters
        self.N = 1000               # number of households
        self.N_min = 1000           # absolute minimum number of households
        self.N0 = self.N            # original number of households -- used in plotting of mean B and mean U
        
        # dwelling parameters
        self.M = 3*self.N           # number of dwellings
        self.u0_max = 5             # general dwelling utility U0
        self.u0_wid = 7             # characteristic width of U0 window for preferred area
        self.price_scale = 1000     # rough scale of $/quarter for pricing methods of dwellings.
        self.M_max = 1.2            # max no. dwellings M permitted (during dwelling creation)

    def create(self):
        """ 
        create() instantiates all objects/classes within the model: 
        households, dwellings, calculations, visualisations.

        """

        global calcs, pop, dwell, visu      # instantiate all objects within the model
        calcs = calculations()
        pop = household(self.N, self.N_min)
        dwell = dwelling(self.M, self.R, self.u0_max, self.u0_wid, self.price_scale, self.M_max)
        visu = visualisation(sim, pop, dwell, calcs)
        
    def load_from_save(self): 
        """
        load_from_save() loads all saved data from a specified save folder and fills this data
        into all profile arrays.

        """

        print("Ensure files to load are unique in directory 'data/input/files...'")
        
        # get files in directory
        files = sorted([filenames for _,_,filenames in os.walk(dirinput)][0])
        files = [x for x in files if 'txt' in x]
        pop_files = [x for x in files if 'household' in x]
        dwell_files = [x for x in files if 'dwelling' in x]
        order = []

        for x in pop_files:
            w = [pos for pos, char in enumerate(x) if char == '-']
            order.append(float(x[w[0]+1:-3]))
        files = np.array(sorted(zip(pop_files,dwell_files,order), key=lambda x:x[2]))
        pop_files = files[:,0]
        dwell_files = files[:,1]

        # get simulation ID and create folder for visualisation output
        w = [pos for pos, char in enumerate(pop_files[0]) if char == '_']
        self.sim_id = pop_files[0][w[0]+1:w[1]]
        self.current_folder = '{}-{}-{}'.format(self.sim_id[4:6],self.sim_id[2:4], self.sim_id[:2])
        self.sim = 'market_dyn'

        # create a directory for simulation output
        chk_dir = os.path.isdir(str(diroutput) + self.current_folder)
        if chk_dir == False:
            os.mkdir(str(diroutput) + self.current_folder)
        self.sim_folder = '{}#{}'.format(self.sim, self.sim_id)
        chk_dir = os.path.isdir(str(diroutput) + self.current_folder+'/'+self.sim_folder)
        if chk_dir == False:
            os.mkdir(str(diroutput)+self.current_folder+'/'+self.sim_folder)

        self.save_loc = str(diroutput) + str(self.current_folder) + '/' + str(self.sim_folder) + '/' # save directory

        # load households & restructure at all timesteps into self.all_store_pop
        for x in pop_files:
            pop = np.loadtxt(str(dirinput)+x, dtype=float)
            struct = pop[0]
            # set sim.p/self.p to household data at last timestep
            self.p = pop[1:].reshape(int(struct[0]),int(struct[1]), int(struct[2]))
            # add to all_store
            self.all_store_pop.append(np.array(self.p, dtype=object))

        #load dwellings & restructure at all timesteps into self.all_store_dwell
        for x in dwell_files:
            dwell = np.loadtxt(str(dirinput)+x, dtype=float)
            struct = dwell[0]
            # set sim.r/self.r to dwelling data at last timestep
            self.r = dwell[1:].reshape(int(struct[0]),int(struct[1]), int(struct[2]))
            self.all_store_dwell.append(np.array(self.r, dtype=object))

        # update the system size
        self.R = int(np.max(self.r[:,1,0]))

    def visualise_results(self):
        """ 
        visualise_results() takes a user input for the types of models to plot,
            i.e. those of the configuration of the model prior to any evaluation (early figures in paper),
            OR those of the data output from a previous simulation (results/test cases in paper).

        """
        print("loading data from previous simulations")

        # load the saved data to begin simulation, only the parameters for length of simulation will remain, 
        # all parameters generated during instantiation of objects will be overwritten with the saved parameters.
        simulate.load_from_save(self)

        """ Case A - City formation """
        visu.spatial_gaussian()
        
        """ Case B - Effect of U0 """
        # visu.densitymap()           # figure 9a-f
        # visu.densitymap_whole()     # figure 8b
        # visu.meanB_r_10()           # figure 10a
        # visu.U_contribution()
        # if hasattr(dwell, 'U0_store'):
        #     visu.U0_dist_3d()        # U0 distribution generated (3d plot)
        #     visu.U0_dist_2d()        # U0 distribution generated (2d plot as function of r) - figure in paper
        # visu.U0_sample_mean()        # mean U0 distribution from sample
        # visu.U0_sample()             # U0 distribution from sample  

        """ Case C - Effects of Spatial Localization of Favored Dwelling Characteristics """
        # visu.B1_dist()
        # visu.C_gaussian_map()   # figure 11b,c: gaussian smoothened spatial density of households
        # visu.C_heatmap()        # greyscale heatmap of spatial density of households

        """ Case D - Segregation """
        # visu.spatial_types()    # figure 12,13 from paper; spatial plot of households by characteristic type q1
        # visu.gaussian_types()   # gaussian filter of density of households by characteristic type q1

        """ Case E - Supply and Demand """
        # visu.N_t()                     # figure 14a: no. households N vs time t
        # visu.meanB_t_r()               # figure 14b: plot mean B at specific radiuses for all t  
        # visu.meanB_r_t()               # figure 14c: plot of mean B at all radiuses for each t
        # visu.mean_density()            # figure 14dL plot of mean density at all radius for each t
        # visu.M_t()                     # no. dwellings M vs time t

        """ extra visualisation functions """
        """ Household related plots """
        # visu.spatial_dist()            # general scatter plot of households + dwellings or KDE of households
        # visu.kdemap()                  # KDE map of households
        # visu.wealthmap()               # spatial map of wealth distribution
        # visu.U_all()                   # scatter U of all households by s0 & line of best fit
        # visu.plot_U_wealthy()          # plot U of top 50 wealthiest households
        # visu.U_specific_household()    # plot utility of specific household
        # visu.U_by_class()              # plot utility of households over time

        """ Dwelling related plots """
        # visu.B0_dwell()                # price B0 of specified dwelling over time
        # visu.B0_household()            # price of B0 for a specified household over time
        # visu.scatter_unocc()           # blue & red scatter plot of B0 over time for all dwellings
        # visu.mean_B0_U()               # plot the mean B0 vs t and mean U vs t
        # visu.B0_M_animate()            # animation - sorted B0 scattered against dwelling no M and s0 on right axis

        # plt.savefig(sim.save_loc + '/plot'+ sim.sim_id +'.png', dpi=400) -- sample line for saving plot
        # visu.liveplot()               # animated plot of the spatial distribution
        # visu.interactive_spatial()    # interactive plot of spatial distribution of households/dwellings

        plt.show()

    def saving(self, timestep):
        # save all vectors related to the household and the dwelling as separate files within specific datetime folder of iteration. 
        # save all model parameters also; model parameter file will include runtime also.   -- NEED TO ADD THIS IN
        # These will be saved in a format such that it can be input as starting data

        print('SAVING DATA')
        self.save_loc = str(diroutput) + str(self.current_folder) + '/' + str(self.sim_folder) + '/' # save directory

        """ compile the household profile array for saving"""
        mn = max([len(pop.p[0])+1, len(pop.q[0]), len(pop.s[0])])
        self.p = np.zeros((pop.N, 4, mn), dtype=object)    # initialise the empty array of the household profile array
        p_dim = np.array(np.shape(self.p))
        p_dim = np.array([np.pad(p_dim, (0,mn-len(p_dim)), 'constant')])
        
        for n, [p, U, q, s] in enumerate(zip(pop.p[0:], pop.U[0:], pop.q[0:], pop.s[0:])):
            self.p[n][0][:len(p)] = np.array(p)       # add general household information
            self.p[n][0][len(p)]  = np.array(U)      # add utility of household
            self.p[n][1][:len(q)] = np.array(q)       # add household characteristics
            self.p[n][2][:len(s)] = np.array(s)       # add price sensitivity & preference strengths of the household

#         ## save household profile vector -- this is just turned off for testing
#         with open(self.save_loc +'households_' + self.sim_id + '_timestep-{}.txt'.format(timestep) , 'w') as save:
#             save.write('# household profile vector shape (no. households, 4, no. preferences/max length)\n')
#             np.savetxt(save, p_dim, fmt='%-7.2f')
#             save.write("""# KEY:
# # [[Household ID, on-market flag, dwelling_id, utility, 0...],
# # [characteristics: q_0(p), q_1(p), q_2(p), ..., q_m(p)],
# # [housing preferences: [s_0(p), s'(p)] = [s_0(p), s'_1(p), s'_2(p), ..., s_n(p)] ]\n""")

#             for data_slice in self.p:
#                 save.write('\n# household {} \n'.format(data_slice[0][0]))
#                 np.savetxt(save, data_slice, fmt='%-7.2f')
        
        """ compile the dwelling profile array for saving """
        r_mn = max([len(dwell.style), len(pop.s_bound)])    # max of length of characteristics B or length of household profile array
        self.r = np.zeros((dwell.M, 3, r_mn), dtype=object)   # initialise the empty array dimensions to save dwelling data
        r_dim = np.array(np.shape(self.r))
        r_dim = np.array([np.pad(r_dim, (0,r_mn-len(r_dim)), 'constant')])

        for n, [r, r_coord, B] in enumerate(zip(dwell.r[0:], dwell.r_coord[0:], dwell.B[0:])):
            self.r[n][0][:len(r)]       = np.array(r)        # assign general information of dwelling
            self.r[n][1][:len(r_coord)] = np.array(r_coord)    # assign dwelling location
            self.r[n][2][:len(B)]       = np.array(B)        # assign dwelling characteristics
            
#         ## save dwelling profile vector -- this is just turned off for testing
#         with open(self.save_loc +'dwellings_' + self.sim_id + '_timestep-{}.txt'.format(timestep) , 'w') as save:
#             save.write('# dwellings profile vector shape (no. dwellings, 3, no. characteristics/max length)\n')
#             np.savetxt(save, r_dim, fmt='%-7.2f')
#             save.write("""# KEY:
# # [[Dwelling ID, on-market flag, U_0(r), 0, ...],
# # [location: r_1(r), r_2(r), 0, 0, 0, 0, ...],
# # [characteristics: B_0(r), B_1(r), ..., B_n(r)]]\n""")

#             for data_slice in self.r:
#                 save.write('\n# dwelling {} \n'.format(data_slice[0][0]))
#                 np.savetxt(save, data_slice, fmt='%-7.2f')

        self.all_store_pop.append(np.array(self.p, dtype=object))
        self.all_store_dwell.append(np.array(self.r, dtype=object))

        print('HOUSEHOLD & DWELLING PROFILE VECTORS SAVED')

class calculations:
    def __init__(self):
        self.fees = 0.00    # percentage of price of dwelling which will be charged as fees for transaction

    def mse_dist(self, locs):
        locs = np.array(locs*10, dtype=int)
        centric = np.array(dwell.centric*10, dtype=int)
        dists = np.linalg.norm(locs.T - centric.T, axis=0)/10
        mse = np.mean(dists**2)
        return mse

class visualisation:
    def __init__(self, sim_in, pop_in, dwell_in, calcs_in):
        self.tit_cnt = 0
        self.rad_t = []
        global sim, pop, dwell, calcs
        sim = sim_in
        pop = pop_in
        dwell = dwell_in
        calcs = calcs_in
    
    def figure_2(self):
        """
        figure_2() creates figures 2a and 2b from the paper:
        - figure 2a: U(x, p_j) of household p_j vs dwelling position x, 
        - figure 2b: spatial varisation of B_1(x) vs x

        this function is mostly hard set via tinkering; just example of distributions

        """
        
        B = dwell.B
        fig, axs = plt.subplots(2,1,figsize=(7, 10), dpi=90)

        """ figure 2a """
        xs = np.linspace(-2, 2.3, 100)
        y2b = -xs
        xs_1 = np.linspace(-1.6, -0.96, 20)
        xs_2 = np.linspace(1.61, 2.13, 20)

        y1 = (xs + np.pi)*np.sin(xs + np.pi)
        y1_s = -5*(xs_1 + 1.3)**2 + 2.35
        a = np.array([-np.sin(x**2/5) for x in xs if x<0])
        b = np.array([np.sin(x**2/2) for x in xs if x>= 0])
        y2 = np.concatenate((a, b))
        y2_s = -5*(xs_2 - 1.9)**2 + 1.4
        y3 = -((xs/0.6 - 2.9)*np.sin(xs/0.6 - 2.9))/4 + 0.17

        y1 = y3 + y2b/2
        y2 = y3 - y2b/2

        axs[0].set_title("(a)", fontname='times', fontsize=20,loc='right')
        axs[0].plot(xs, y1, 'black', linestyle='--')
        axs[0].plot(xs_1, y1_s, 'black', linestyle='dotted')
        axs[0].plot(xs, y2, 'black', linestyle='--')
        axs[0].plot(xs_2, y2_s, 'black', linestyle='dotted')
        axs[0].plot(xs, y3, 'black')
        axs[0].set_xlim([-2, 2.3])
        axs[0].set_ylim([-1.5, 2.6])
        axs[0].text(-1.9, 1.2, r'$\rm p_{1}$',  fontdict={'family':'times','weight': 'bold','size': 16})
        axs[0].text(-1.9, -0.8, r'$\rm p_{2}$',  fontdict={'family':'times','weight': 'bold','size': 16})

        axs[0].set_xlabel(r"$x$", fontsize=18, fontname='times new roman')
        axs[0].set_ylabel(r"$U(x, p_j)$", fontsize=18, fontname='times new roman')
        axs[0].set_xticks(ticks=[])
        axs[0].set_xticklabels(labels=[])
        axs[0].set_yticks(ticks=[])
        axs[0].set_yticklabels(labels=[])

        """ figure 2b """
        xs = np.linspace(-2, 2.3, 100)
        y2b = -xs

        axs[1].set_title("(b)", fontname='times', fontsize=20,loc='right')
        axs[1].plot([xs[0], xs[-1]],[0, 0], 'black', linestyle='--' )
        axs[1].plot(xs, y2b,  'black')
        axs[1].set_xlabel(r"$x$", fontsize=18, fontname='times new roman')
        axs[1].set_ylabel(r"$B_1(x)$", fontsize=18, fontname='times')
        axs[1].set_xticks(ticks=[])
        axs[1].set_xticklabels(labels=[])
        axs[1].set_yticks(ticks=[0])
        axs[1].set_yticklabels(labels=[0])
        for tick in axs[1].get_yticklabels():
            tick.set_fontname("times")
            tick.set_fontsize(14)

        axs[1].set_xlim([-2, 2.3])
        axs[1].set_ylim([-2.3, 2])
        # plt.savefig(sim.save_loc + '/figure_2-'+ sim.sim_id +'.png', dpi=400)

    def figure_3(self):
        """
        figure_3() creates figure 3 from the paper, utility function vs 1D position 
        for varying s0 values of a household.

        this function is mostly hard set via tinkering; just example of distribution.
        """

        x = np.linspace(0, 2.5, num=1000)
        y0 = np.sin(x**2)/5 +1
        y1 = np.sin(x**2)/5 +1.2

        plt.figure(1, tight_layout=True, figsize=[5.4,3.8])
        plt.plot(x, y0, 'black',linestyle='--')
        plt.plot(x, y1,'black')
        plt.ylim([0.6,1.7])
        plt.xlim([0, 2.5])
        plt.xticks(ticks=[], labels=[])
        plt.yticks(ticks=[], labels=[])

        plt.xlabel(r"$x$", fontsize=18, fontname='times new roman')
        plt.ylabel(r"$U(x,p_j)$", fontsize=18, fontname='times')
        # plt.savefig(sim.save_loc + '/figure_3-'+ sim.sim_id +'.png', dpi=400)

    def figure_4abc(self):
        """
        figure_4abc() creates figues 4a, 4b and 4c from the paper:
        - figure 4a: log(F(m)) vs log(m) from eq. 11, star at knee at m_c.
        - figure 4b: log(F(s_0)) vs s0 from eq. 12.
        - figure 4c: m vs s0

        use figsize 6,13 for same fig size as in paper & save as img
        use figsize 5.3,10 for fig that fits screen.
        """

        fig, axs = plt.subplots(3,1,figsize=(5.3, 10), dpi=90,tight_layout=True)
        
        """ figure 4a """
        axs[0].plot(np.log(calcs.m), np.log(calcs.F_m), c='black')
        w = np.where(np.log(calcs.F_m)==np.log(calcs.m_c))

        axs[0].scatter([np.log(calcs.m_c)],[-4.75],marker='*',s=100, c='black')
        axs[0].scatter([np.min(np.log(calcs.m)), np.max(np.log(calcs.m))],[np.max(np.log(calcs.F_m)), np.min(np.log(calcs.F_m))],s=100, marker='.',c='black')
        axs[0].text(9.3, -4.6, r'$ F(m_\mathrm{min})$',  fontdict={'family':'times','weight': 'bold','size': 16})
        axs[0].text(11.8, -7.4, r'$ F(m_\mathrm{max})$',  fontdict={'family':'times','weight': 'bold','size': 16})
        axs[0].text(10.95, -4.75, r'$ F(m_{c})$',  fontdict={'family':'times','weight': 'bold','size': 16})

        axs[0].set_xlabel(r'$ \log_{10} \> m$', fontsize=18)
        axs[0].set_ylabel(r'$ \log_{10}\>F(m)$',fontsize=18)
        for tick in axs[0].get_xticklabels():
            tick.set_fontname("times")
            tick.set_fontsize(14)
        for tick in axs[0].get_yticklabels():
            tick.set_fontname("times")
            tick.set_fontsize(14)
        axs[0].set_title("(a)", fontname='times', fontsize=20,loc='right')

        """ figure 4b """
        axs[1].plot(calcs.s_0, np.log(calcs.F_s), c='black')
        axs[1].set_xlabel(r'$ s_0$',fontsize=18)
        axs[1].set_ylabel(r'$ \log_{10} \>F(s_0)$',fontsize=18)
        axs[1].set_title("(b)", fontname='times', fontsize=20,loc='right')
        for tick in axs[1].get_xticklabels():
            tick.set_fontname("times")
            tick.set_fontsize(14)
        for tick in axs[1].get_yticklabels():
            tick.set_fontname("times")
            tick.set_fontsize(14)
        axs[1].scatter([np.min(calcs.s_0), np.max(calcs.s_0)],[np.max(np.log(calcs.F_s)), np.min(np.log(calcs.F_s))],s=100, marker='.',c='black')
        axs[1].text(-0.97, 10.3, r'$ s_0(m_\mathrm{min})$',  fontdict={'family':'times','weight': 'bold','size': 16})
        axs[1].text(-0.2, -1.6, r'$ s_0(m_\mathrm{max})$',  fontdict={'family':'times','weight': 'bold','size': 16})

        """ figure 4c """
        axs[2].plot(calcs.s_0, calcs.m, c='black')
        axs[2].set_xlabel(r'$ s_0$',fontsize = 18)
        axs[2].set_ylabel(r'$ m \>\>(\$10^{4}) $',fontsize=18)
        axs[2].set_title("(c)", fontname='times', fontsize=20,loc='right')
        axs[2].scatter([np.min(calcs.s_0), np.max(calcs.s_0)],[np.min(calcs.m), np.max(calcs.m)],s=100, marker='.',c='black')
        axs[2].set_yticks(ticks=np.linspace(min(calcs.m), max(calcs.m), 6, dtype=int))
        axs[2].set_yticklabels(labels=np.linspace(min(calcs.m), max(calcs.m), 6, dtype=int)//10000)
        axs[2].text(-0.97, 13000, r'$ m(s_{0 \mathrm{min}})$',  fontdict={'family':'times','weight': 'bold','size': 16})
        axs[2].text(-0.15, 235000, r'$ m(s_{0 \mathrm{max}})$',  fontdict={'family':'times','weight': 'bold','size': 16})
        for tick in axs[2].get_xticklabels():
            tick.set_fontname("times")
            tick.set_fontsize(14)
        for tick in axs[2].get_yticklabels():
            tick.set_fontname("times")
            tick.set_fontsize(14)
        # plt.savefig(sim.save_loc + '/figure_4abc-'+ sim.sim_id +'.png', dpi=400)

    def figure_4a(self):
        """ 
        *** NOT UPDATED PLOT ****
        Figure 4a from paper with dashed line at m_c
        """

        m = calcs.m
        F_m = calcs.F_m
        m_c = calcs.m_c

        plt.figure()
        plt.plot(np.log(m), np.log(F_m), c='black')
        plt.plot([np.log(m_c), np.log(m_c)],[min(np.log(F_m)), max(np.log(F_m))], c='grey',linestyle='--')
        plt.xlabel('$m$ (log scale)', fontsize=12)
        plt.ylabel('$F(m)$ (log scale)',fontsize=12)
        plt.legend(['Income distribution $F(m)$', 'Income drop-off $m_c$'],fontsize=12)
        # plt.savefig(sim.save_loc + '/figure_4a-'+ sim.sim_id +'.png', dpi=400)
    
    def figure_4b_normalised(self):
        """ 
        *** NOT UPDATED PLOT ****
        !!! NOT USED IN PAPER !!!

        figure_4b_normalised() plots normalised version of figure 4b from paper; normalised to 1.

        - this is effectively the probability distribution in which the household price sensitivity
        is then drawn from.
        """

        s_0 = calcs.s_0
        F_s_norm = calcs.F_s_norm

        plt.figure()
        plt.title('Figure 4b (normalized)')
        plt.plot(s_0, F_s_norm)
        plt.xlabel('$s_0$')
        plt.ylabel('Normalized $F(s_0)$')
        # plt.savefig(sim.save_loc + '/figure_4b_normalised-'+ sim.sim_id +'.png', dpi=400)
    
    def m_s0_p(self):
        """ 
        *** NOT UPDATED PLOT ****
        !!! NOT USED IN PAPER !!!

        m_s0_p() plots household income m(p) (left) and price sensitivity s_0(p) vs household p.
        """

        m = calcs.m_sampled
        s_0 = calcs.s_0_sampled

        fig = plt.figure(tight_layout=True)
        ax = fig.add_subplot(111)
        ax.scatter(np.linspace(1,len(m),len(m)),m,marker='.',s=10)
        ax2 = ax.twinx()
        # ax2.scatter(np.linspace(1,len(m),len(m)),s_0,marker='.',s=0)
        ax2.set_yticks(ticks=np.linspace(0,1,6, dtype=float))
        ax2.set_yticklabels(labels=np.round(np.linspace(np.min(s_0),np.max(s_0),6, dtype=float),2), rotation=0)
        ax.set_xlabel(r'Household $p$')
        ax.set_ylabel(r'Income $m(p)$')
        ax2.set_ylabel(r'Price sensitivity $s_0(p)$')
        # plt.savefig(sim.save_loc + '/m_s0_p-'+ sim.sim_id +'.png', dpi=400)

    def s0_N(self):
        """ 
        *** NOT UPDATED PLOT ****
        !!! NOT USED IN PAPER !!!

        s0_N() plots price sensitivity s0 vs household N.
        """

        draw = calcs.draw

        plt.figure(tight_layout=True)
        plt.title('Distribution of $s_0$ by household $N$')
        plt.scatter(range(len(draw)),sorted(draw), marker='.',s=10)
        plt.xlabel('Household $N$')
        plt.ylabel('$s_0$')
        # plt.savefig(sim.save_loc + '/s0_N-'+ sim.sim_id +'.png', dpi=400)

    def figure_5(self):
        """
        figure_5() creates figure 5 from the paper, utility function U(x,p_j,t) 
        and price function P(x,t) over all p at time t.

        this function is mostly hard set via tinkering of sine curves;
        - just example of distribution.
        """

        xs = np.linspace(2.6,3.6, 1001)
        u0 = np.array([-(np.sin(xs**2)*np.cos(xs))/10 + 4]).T
        u0_max = -(np.sin(3.13**2)*np.cos(3.13))/10 + 4

        u1 = np.array([-(1.5*np.sin(xs**2+1.4)*np.cos(2*xs))/5 + 4]).T
        u1_min = -(1.5*np.sin(2.81**2+1.4)*np.cos(2*2.81))/5 + 4
        u1_max = -(1.5*np.sin(3.6**2+1.4)*np.cos(2*3.6))/5 + 4

        u2 = np.array([-(np.sin(xs**2)*np.cos(2*xs))/5 + 4]).T
        u2_min = -(np.sin(2.99**2)*np.cos(2*2.99))/5 + 4
        u2_max = -(np.sin(3.6**2)*np.cos(2*3.6))/5 + 4

        u3 = np.array([-(np.sin(xs**2-1.2)*np.cos(2*xs))/5 + 4]).T
        u3_min = -(np.sin(3.2**2-1.2)*np.cos(2*3.2))/5 + 4

        u0[np.where(u0==u0_max)[0][0]:] = 0
        u1[:np.where(u1==u1_min)[0][0]] = 0
        u1[np.where(u1==u1_max)[0][0]:] = 0
        u2[:np.where(u2==u2_min)[0][0]] = 0
        u2[np.where(u2==u2_max)[0][0]:] = 0
        u3[:np.where(u3==u3_min)[0][0]] = 0

        price = np.concatenate((u0, u1, u2, u3), axis=1)
        price = np.max(price, axis=1)
        
        u0 = u0[np.nonzero(u0)]
        x0 = np.linspace(2.6,3.13,len(u0))
        u1 = u1[np.nonzero(u1)]
        x1 = np.linspace(2.81,3.6,len(u1))
        u2 = u2[np.nonzero(u2)]
        x2 = np.linspace(2.99,3.6,len(u2))
        u3 = u3[np.nonzero(u3)]
        x3 = np.linspace(3.2,3.6,len(u3))
        
        plt.figure(tight_layout=True, figsize=[5.4,3.8])
        plt.plot(x0, u0, 'grey',linestyle='dotted')
        plt.plot(x1, u1, 'grey', linestyle='dotted')
        plt.plot(x2, u2, 'grey',linestyle='dotted')
        plt.plot(x3, u3, 'grey',linestyle='dotted')

        plt.plot(xs, price, 'black')
        plt.xticks(ticks=[], labels=[])
        plt.yticks(ticks=[], labels=[])

        plt.xlabel(r"$x$", fontsize=18)
        ylab = r"$U(x,p_j,t), \>\>\> P(x,t)$"
        plt.ylabel(str(ylab), fontsize=18)
        plt.ylim([3.97, 4.35])
        plt.xlim([np.min(x0), np.max(x3)])
        # plt.savefig(sim.save_loc + '/figure_5-'+ sim.sim_id +'.png', dpi=400)
    
    def spatial_gaussian(self):
        """
        CASE A - figure 7a-h

        left column: spatial plots of households & dwellings
        right column: gaussian smoothened plots of household distribution

        caseA25 gives figures 7a-f
        caseA50 gives figures 7g-h & gif creation
        """

        yrs = np.array([0, 10, 25])*4
        yrs = yrs[yrs<=sim.t*sim.t_inc]

        for yr in yrs:
            yr = int(yr)

            dwell_r = np.array(sim.all_store_dwell[yr][:,0], dtype=float)
            dwell_rcoord = np.array(sim.all_store_dwell[yr][:,1,[0,1]], dtype=float)
            X_r, Y_r = dwell_rcoord[:,0:2].T

            pop_p = np.array(sim.all_store_pop[yr][:,0],dtype=float)
    
            types = [len(pop_p)]
            self.types = types
            colours = ['Black','crimson','forestgreen'] #'darkorange'

            self.edge = ['lightgrey']*len(X_r)
            self.c = [0]*len(X_r)

            for y, x in enumerate(types):
                self.edge.extend([colours[y]]*x)
                self.c.extend([y+1]*x)            

            # cmap = ['lightgrey'] + colours[:len(types)]   # dwellings in light grey
            cmap = ['crimson'] + colours[:len(types)]   # dwellings in light crimson
            leg = ['Dwellings', 'Households']
            self.cmap = ListedColormap(cmap)
            
            X_p, Y_p = dwell_rcoord[:,0:2][np.array(pop_p[:,2],dtype=int)].T    # coordinates of each household
            self.loc = dwell_rcoord[:,0:2][np.array(pop_p[:,2],dtype=int)].T
            self.X_p, self.Y_p = X_p, Y_p

            # household of interest
            # houseX, houseY = sim.r[:,1,0:2][np.array(sim.p[np.where(sim.p[:,0,0]==576),0,2],dtype=int)][0][0]

            self.X_c = np.concatenate((X_r, X_p), axis=0)
            self.Y_c = np.concatenate((Y_r, Y_p), axis=0)
            self.s = [10]*len(X_r) + [10]*len(X_p)
            self.mark = [str(".")]*len(X_r) + [str(".")]*len(X_p)
            
            if sim.full_case == 'A':
                titles = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s']
            elif sim.full_case == 'A50':
                titles = ['a','b','c','d','g','h','g','h','i','j','k','l','m','n','o','p','q','r','s']

            fig, axs = plt.subplots(1,2)
            plt.rcParams['mathtext.fontset'] = 'stix'
            plt.rcParams['font.family'] = 'STIXGeneral'
            axs[0].scatter(X_r, Y_r, s=[10]*len(X_r), c=[0]*len(X_r), edgecolors=['crimson']*len(X_r), cmap=ListedColormap('crimson'), marker='.', alpha=0.2)
            axs[0].scatter(X_p, Y_p, s=[10]*len(X_p), c=[1]*len(X_p), edgecolors=['black']*len(X_p), cmap=ListedColormap('Black'), marker='.', alpha=1)

            axs[0].set_xlabel("{} (km)".format(r'$x$'), fontsize=18, fontname='times')
            axs[0].set_ylabel("{} (km)".format(r'$y$'), fontsize=18, fontname='times')
            lims = 0.02*sim.R
            axs[0].set_xlim([-lims, sim.R+lims])
            axs[0].set_ylim([-lims, sim.R+lims])
            axs[0].set_xticks(ticks=np.linspace(0,sim.R,6))
            axs[0].set_yticks(ticks=np.linspace(0,sim.R,6))
            for tick in axs[0].get_xticklabels():
                tick.set_fontname("times")
                tick.set_fontsize(14)
            for tick in axs[0].get_yticklabels():
                tick.set_fontname("times")
                tick.set_fontsize(14)
            axs[0].set_aspect('equal', adjustable='box')
            axs[0].set_title("({})".format(titles[self.tit_cnt]), fontdict={'family':'times','weight': 'bold','size': 24},loc='right')
            self.tit_cnt += 1

            # GAUSSIAN SMOOTHENING PLOT
            inc = 0.25
            dim = int(int(sim.R//inc + 1))
            locs = np.zeros(shape=(dim,dim))
            for x, y in zip(self.X_p, self.Y_p):
                locs[int(x*4)][int(y*4)] += 1
            sig = 3
            ax = 1
            locs1 = gaussian_filter(locs, sigma=sig)

            C = locs1.T
            img = axs[ax].imshow(C, cmap='Greys', interpolation='nearest', vmin=0, vmax=0.5)
            axs[ax].set_xlabel("{} (km)".format(r'$x$'), fontsize=18, fontname='times')
            axs[ax].set_ylabel("{} (km)".format(r'$y$'), fontsize=18, fontname='times')
            for tick in axs[ax].get_xticklabels():
                tick.set_fontname("times")
                tick.set_fontsize(14)
            for tick in axs[ax].get_yticklabels():
                tick.set_fontname("times")
                tick.set_fontsize(14)

            lims = 0.02*sim.R
            axs[ax].set_xlim([-lims, sim.R+lims])
            axs[ax].set_ylim([-lims, sim.R+lims])
            axs[ax].set_xticks(ticks=np.linspace(0,sim.R*4+1,6, dtype=int))
            axs[ax].set_xticklabels(labels=np.linspace(0,sim.R,6, dtype=int), rotation=0)
            axs[ax].set_yticks(ticks=np.linspace(0,sim.R*4+1,6, dtype=int))
            axs[ax].set_yticklabels(labels=np.linspace(0,sim.R,6, dtype=int))
            axs[ax].set_aspect('equal', adjustable='box')
            axs[ax].set_title("({})".format(titles[self.tit_cnt]), fontdict={'family':'times','weight': 'bold','size': 24},loc='right')
            
            # plt.colorbar(img, ax=axs)
            # axs[ax].invert_yaxis()
            self.tit_cnt += 1

            # # KDE PLOT INSTEAD OF GAUSSIAN
            # sns.kdeplot(self.loc, cmap='Greys', shade=True, bw=2, ax=axs[1])
            # axs[1].set_xlabel('x (km)',fontsize=16, fontname='times')
            # axs[1].set_ylabel('y (km)',fontsize=16, fontname='times')
            # for tick in axs[1].get_xticklabels():
            #     tick.set_fontname("times")
            #     tick.set_fontsize(12)
            # for tick in axs[1].get_yticklabels():
            #     tick.set_fontname("times")
            #     tick.set_fontsize(12)
            # axs[1].set_xlim([-1, sim.R+1])
            # axs[1].set_ylim([-1, sim.R+1])
            # axs[1].set_xticks(ticks=np.linspace(0,sim.R,6))
            # # axs[0].set_xticklabels(labels=[0,5,10,15,20,25])
            # axs[1].set_yticks(ticks=np.linspace(0,sim.R,6))
            # # axs[0].set_yticklabels(labels=[0,5,10,15,20,25])
            # axs[1].set_aspect('equal', adjustable='box')
            # axs[1].set_title("({})".format(titles[self.tit_cnt]), fontdict={'family':'times','weight': 'bold','size': 20},loc='right')
            # # plt.title('Year {}'.format(self.tit_cnt), x=-0.5, y=1)
            # self.tit_cnt += 1
            plt.tight_layout()    
            plt.savefig(sim.save_loc + '/urbanfig7{}{}'.format(titles[self.tit_cnt-2],titles[self.tit_cnt-1]) +'.png', dpi=400)
    
    def spatial_gaussian_movie(self):
        """
        CASE A - figure 7g gif creation

        spatial plot of households & dwellings

        caseA50 gives figures 7g-h & gif creation
        """

        yrs = np.arange(0,101,dtype=int)    # this is for movie creation -- save all figures to folder then use
        # https://ezgif.com/maker to compile pics into .gif
        yrs = yrs[yrs<=sim.t*sim.t_inc]
        for yr in yrs:
            yr = int(yr)

            dwell_r = np.array(sim.all_store_dwell[yr][:,0], dtype=float)
            dwell_rcoord = np.array(sim.all_store_dwell[yr][:,1,[0,1]], dtype=float)
            X_r, Y_r = dwell_rcoord[:,0:2].T

            pop_p = np.array(sim.all_store_pop[yr][:,0],dtype=float)
    
            types = [len(pop_p)]
            self.types = types
            colours = ['Black','crimson','forestgreen'] #'darkorange'

            self.edge = ['lightgrey']*len(X_r)
            self.c = [0]*len(X_r)

            for y, x in enumerate(types):
                self.edge.extend([colours[y]]*x)
                self.c.extend([y+1]*x)            

            # cmap = ['lightgrey'] + colours[:len(types)]   # dwellings in light grey
            cmap = ['crimson'] + colours[:len(types)]   # dwellings in light crimson
            leg = ['Dwellings', 'Households']
            self.cmap = ListedColormap(cmap)
            
            X_p, Y_p = dwell_rcoord[:,0:2][np.array(pop_p[:,2],dtype=int)].T    # coordinates of each household
            self.loc = dwell_rcoord[:,0:2][np.array(pop_p[:,2],dtype=int)].T
            self.X_p, self.Y_p = X_p, Y_p

            # household of interest
            # houseX, houseY = sim.r[:,1,0:2][np.array(sim.p[np.where(sim.p[:,0,0]==576),0,2],dtype=int)][0][0]

            self.X_c = np.concatenate((X_r, X_p), axis=0)
            self.Y_c = np.concatenate((Y_r, Y_p), axis=0)
            self.s = [10]*len(X_r) + [10]*len(X_p)
            self.mark = [str(".")]*len(X_r) + [str(".")]*len(X_p)
            
            if sim.full_case == 'A':
                titles = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s']
            elif sim.full_case == 'A50':
                titles = ['a','b','c','d','g','h','g','h','i','j','k','l','m','n','o','p','q','r','s']


            fig, axs = plt.subplots(1)
            plt.rcParams['mathtext.fontset'] = 'stix'
            plt.rcParams['font.family'] = 'STIXGeneral'
            axs.scatter(X_r, Y_r, s=[10]*len(X_r), c=[0]*len(X_r), edgecolors=['crimson']*len(X_r), cmap=ListedColormap('crimson'), marker='.', alpha=0.2)
            axs.scatter(X_p, Y_p, s=[10]*len(X_p), c=[1]*len(X_p), edgecolors=['black']*len(X_p), cmap=ListedColormap('Black'), marker='.', alpha=1)

            axs.set_xlabel("{} (km)".format(r'$x$'), fontsize=18, fontname='times')
            axs.set_ylabel("{} (km)".format(r'$y$'), fontsize=18, fontname='times')
            lims = 0.02*sim.R
            axs.set_xlim([-lims, sim.R+lims])
            axs.set_ylim([-lims, sim.R+lims])
            axs.set_xticks(ticks=np.linspace(0,sim.R,6))
            axs.set_yticks(ticks=np.linspace(0,sim.R,6))
            for tick in axs.get_xticklabels():
                tick.set_fontname("times")
                tick.set_fontsize(14)
            for tick in axs.get_yticklabels():
                tick.set_fontname("times")
                tick.set_fontsize(14)
            axs.set_aspect('equal', adjustable='box')
            # axs[0].set_title("({})".format(titles[self.tit_cnt]), fontdict={'family':'times','weight': 'bold','size': 24},loc='right')
            # self.tit_cnt += 1

            # GAUSSIAN SMOOTHENING PLOT
            # inc = 0.25
            # dim = int(int(sim.R//inc + 1))
            # locs = np.zeros(shape=(dim,dim))
            # for x, y in zip(self.X_p, self.Y_p):
            #     locs[int(x*4)][int(y*4)] += 1
            # sig = 3
            # ax = 1
            # locs1 = gaussian_filter(locs, sigma=sig)

            # C = locs1.T
            # img = axs[ax].imshow(C, cmap='Greys', interpolation='nearest', vmin=0, vmax=0.5)
            # axs[ax].set_xlabel("{} (km)".format(r'$x$'), fontsize=18, fontname='times')
            # axs[ax].set_ylabel("{} (km)".format(r'$y$'), fontsize=18, fontname='times')
            # for tick in axs[ax].get_xticklabels():
            #     tick.set_fontname("times")
            #     tick.set_fontsize(14)
            # for tick in axs[ax].get_yticklabels():
            #     tick.set_fontname("times")
            #     tick.set_fontsize(14)

            # lims = 0.02*sim.R
            # axs[ax].set_xlim([-lims, sim.R+lims])
            # axs[ax].set_ylim([-lims, sim.R+lims])
            # axs[ax].set_xticks(ticks=np.linspace(0,sim.R*4+1,6, dtype=int))
            # axs[ax].set_xticklabels(labels=np.linspace(0,sim.R,6, dtype=int), rotation=0)
            # axs[ax].set_yticks(ticks=np.linspace(0,sim.R*4+1,6, dtype=int))
            # axs[ax].set_yticklabels(labels=np.linspace(0,sim.R,6, dtype=int))
            # axs[ax].set_aspect('equal', adjustable='box')
            # axs[ax].set_title("({})".format(titles[self.tit_cnt]), fontdict={'family':'times','weight': 'bold','size': 24},loc='right')

            # plt.colorbar(img, ax=axs)
            # axs[ax].invert_yaxis()
            # self.tit_cnt += 1

            # # KDE PLOT INSTEAD OF GAUSSIAN
            # sns.kdeplot(self.loc, cmap='Greys', shade=True, bw=2, ax=axs[1])
            # axs[1].set_xlabel('x (km)',fontsize=16, fontname='times')
            # axs[1].set_ylabel('y (km)',fontsize=16, fontname='times')
            # for tick in axs[1].get_xticklabels():
            #     tick.set_fontname("times")
            #     tick.set_fontsize(12)
            # for tick in axs[1].get_yticklabels():
            #     tick.set_fontname("times")
            #     tick.set_fontsize(12)
            # axs[1].set_xlim([-1, sim.R+1])
            # axs[1].set_ylim([-1, sim.R+1])
            # axs[1].set_xticks(ticks=np.linspace(0,sim.R,6))
            # # axs[0].set_xticklabels(labels=[0,5,10,15,20,25])
            # axs[1].set_yticks(ticks=np.linspace(0,sim.R,6))
            # # axs[0].set_yticklabels(labels=[0,5,10,15,20,25])
            # axs[1].set_aspect('equal', adjustable='box')
            # axs[1].set_title("({})".format(titles[self.tit_cnt]), fontdict={'family':'times','weight': 'bold','size': 20},loc='right')
            # # plt.title('Year {}'.format(self.tit_cnt), x=-0.5, y=1)
            # self.tit_cnt += 1
            plt.tight_layout()    
            plt.savefig(sim.save_loc + '/A_spatial_50_{}-'.format(yr)+ sim.sim_id +'.png', dpi=200)
            plt.close()


    
    def U0_dist_2d(self):
        """
        CASE B - figure 8a

        U0_dist_2d() plots the distribution of U0 in a 2D plot;
        - side-cut from r=0 to r=R of the 3D plot below.
        """

        r = dwell.R/2
        d = np.sqrt(r**2 + r**2)
        x = np.linspace(0,d, 100)
        u0 = dwell.u0_max*np.exp(-abs(x**2/(dwell.u0_wid**2)))
        u0 = u0/max(u0)

        plt.figure(tight_layout=True)
        plt.plot(x, u0, 'Black')
        plt.xticks(ticks=np.linspace(0,d,6), labels=np.linspace(0,d,6,dtype=int))
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.yticks(np.round(np.linspace(0,1,6,dtype=float),2))
        plt.xlabel(r"$r$" + " (km)", fontsize=20, fontname='times new roman')
        plt.ylabel(r"$U_0( r)\>/\>U_{\rm max}$", fontsize=20, fontname='times new roman')
        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['font.family'] = 'STIXGeneral'
        plt.title("(a)", fontname='times', fontsize=28, loc='right')
        plt.xlim([0, d])
        plt.ylim([0, 1])
        plt.savefig(sim.save_loc + '/urbanfig8a.png', dpi=400)

    def U0_dist_3d(self):
        """ 
        CASE B

        U0_dist_3d() plots the distribution of U0 in a 3D plot.
        !!! NOT USED IN PAPER !!!
        """

        U0_store = dwell.U0_store
        incs = dwell.incs
        X,Y = np.meshgrid(np.arange(0,incs,1),np.arange(0,incs,1))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y,U0_store, cmap='viridis')
        c=256
        ax.set_facecolor((c/256,c/256,c/256))
        ax.w_xaxis.set_pane_color((c/256,c/256,c/256, c/256))
        ax.w_yaxis.set_pane_color((c/256,c/256,c/256, c/256))
        ax.grid(False)

        ax.set_xlim([incs,1])
        ax.set_zlim([np.min(U0_store),2.5*np.max(U0_store)])
        ax.w_zaxis.line.set_lw(0.)
        ax.set_zticks([])
        ax.set_ylim([1,incs])
        incs = np.linspace(0,np.max(U0_store), 6)
        ax.set_xticklabels(np.linspace(0,dwell.R,6,dtype=int))
        ax.set_yticklabels(np.linspace(0,dwell.R,6,dtype=int))
        ax.set_xlabel("x (km)", fontsize=18, fontname='times new roman', labelpad=10)
        ax.set_ylabel("y (km)", fontsize=18, fontname='times new roman', labelpad=10)
        for tick in ax.get_xticklabels():
            tick.set_fontname("times")
            tick.set_fontsize(12)
        for tick in ax.get_yticklabels():
            tick.set_fontname("times")
            tick.set_fontsize(12)
        for tick in ax.get_zticklabels():
            tick.set_fontname("times")
            tick.set_fontsize(12)
        plt.title("(a)", fontname='times', fontsize=24, x=1, y=0.8)
        cbaxes = fig.add_axes([0.8, 0.08, 0.02, 0.55])  

        cbar = fig.colorbar(cm.ScalarMappable(cmap='viridis'), ax=ax, shrink=0.2, cax = cbaxes)
        cbar.set_ticks(np.linspace(0,1,5))
        u0_ = r'$U_0$'
        u0_max = r'$U_{max}$'

        # cbar.set_ticklabels(np.round(np.linspace(np.min(self.U0_store),np.max(self.U0_store),5),1))
        cbar.set_ticklabels(np.round(np.linspace(0,1,5),1))
        cbar.set_label('{} / {}'.format(u0_,u0_max), fontsize=16, labelpad=10)
        cbar.ax.tick_params(labelsize=12)
        fig.subplots_adjust(left=0, right=0.8, wspace=0)    # this plus no 'tight_layout=True' allows for adjusting of margins
        # plt.savefig(sim.save_loc + '/B_U0_dist_3d-'+ sim.sim_id +'.png', dpi=400)
    
    def U0_sample_mean(self):
        """
        CASE B 

        U0_sample() creates a 3D plot of the >mean< U0 sampled from the generated U0 distribution.
        !!! NOT USED IN PAPER !!!
        """

        u_0 = dwell.u_0

        locs = np.zeros(shape=(sim.R+1,sim.R+1))
        cnt = np.ones(shape=(sim.R+1,sim.R+1))

        for x, y, z in zip(dwell.r_coord[:,0], dwell.r_coord[:,1], u_0):
            locs[int(np.round(x))][int(np.round(y))] += z
            cnt[int(np.round(x))][int(np.round(y))] += 1
        locs = np.array(locs/cnt)

        X,Y = np.meshgrid(np.arange(0,sim.R+1,1),np.arange(0,sim.R+1,1))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, locs, cmap=cm.coolwarm)
        lims = 0.02*sim.R
        ax.set_xlim([-lims,sim.R+lims])
        ax.set_ylim([-lims,sim.R+lims])
        plt.xlabel("x (km)")
        plt.ylabel("y (km)")
        ax.set_zlabel("mean U0 (1km square sampled i.e. integer coordinate location)")
        # plt.savefig(sim.save_loc + '/B_U0_sample_mean-'+ sim.sim_id +'.png', dpi=400)

    def U0_sample(self):
        """
        CASE B

        U0_sample() creates a 3D plot of the summed U0 sampled from the generated U0 distribution;
        gaussian smoothed.
        !!! NOT USED IN PAPER !!!
        """

        locs = np.zeros(shape=(dwell.R+1,dwell.R+1))
        for x, y, z in zip(dwell.r_coord[:,0], dwell.r_coord[:,1], dwell.u_0):
            locs[int(np.round(x))][int(np.round(y))] += z
        locs = np.array(locs)
        sig=0.5
        smooth = gaussian_filter(locs, sigma=sig)

        X,Y = np.meshgrid(np.arange(0,dwell.R+1,1),np.arange(0,dwell.R+1,1))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, smooth, cmap='viridis')
        ax.set_title('Dwelling U0 sampled from distribution \n(by integer location; smoothed gaussian filter, sigma={})'.format(sig))
        ax.set_xlim([-1,dwell.R+1])
        ax.set_ylim([-1,dwell.R+1])
        # plt.savefig(sim.save_loc + '/B_U0_sample-'+ sim.sim_id +'.png', dpi=400)

    def densitymap_whole(self):
        """
        CASE B - figure 8b

        densitymap() plots the density of households through the use of a gaussian filter
        over incrementalised count.
        """

        fig, axs = plt.subplots(1)
        z = 0

        pop_p = np.array(sim.all_store_pop[-1],dtype=float)
        dwell_rcoord = np.array(sim.all_store_dwell[-1][:,1,[0,1]], dtype=float)
        types = [1,2,3]

        for x in range(len(types)):
            if x == 0 or x == 2:
                if x == 0:
                    # loc = sim.r[:,1,0:2][np.array(sim.p[:,0,2], dtype=int)]
                    loc = dwell_rcoord[:,0:2][np.array(pop_p[:,0,2],dtype=int)]

                ## GAUSSIAN PLOT
                # determine the density of households as by 0.25km increments
                inc = 0.25
                dim = int(int(sim.R//inc + 1))
                locs = np.zeros(shape=(dim,dim))
                X_p = loc[:,0]
                Y_p = loc[:,1]
                for x, y in zip(X_p, Y_p):
                    locs[int(x*4)][int(y*4)] += 1

                sig = 3
                locs = gaussian_filter(locs, sigma=sig)
                C = locs.T
                img = axs.imshow(C, cmap='Greys', interpolation='nearest', vmin=0, vmax=0.4)
                plt.rcParams['mathtext.fontset'] = 'stix'
                plt.rcParams['font.family'] = 'STIXGeneral'

                ## KDE PLOT
                # sns.kdeplot(loc[:, [0,1]], cmap='Greys', shade=True, bw_adjust=.5, ax=axs)
                axs.set_title("(b)", fontdict={'family':'times','weight': 'bold','size': 28},loc='right')
                axs.set_xlabel("{} (km)".format(r'$x$'), fontsize=20, fontname='times')
                axs.set_ylabel("{} (km)".format(r'$y$'), fontsize=20, fontname='times')
                axs.set_xticks(ticks=np.linspace(0,dim,6))
                axs.set_xticklabels(labels=np.linspace(0,sim.R,6, dtype=int), rotation=0)
                axs.set_yticks(ticks=np.linspace(0,dim,6))
                axs.set_yticklabels(labels=np.linspace(0,sim.R,6, dtype=int))
                for tick in axs.get_xticklabels():
                    tick.set_fontname("times")
                    tick.set_fontsize(16)
                for tick in axs.get_yticklabels():
                    tick.set_fontname("times")
                    tick.set_fontsize(16)
                self.tit_cnt += 1
                axs.set_aspect('equal', adjustable='box')
                axs.invert_yaxis()
                # plt.colorbar(img, ax=axs)

                z +=1

        plt.tight_layout()
        plt.savefig(sim.save_loc + '/urbanfig8b.png', dpi=400)

    def densitymap_wealthlevel(self):
        """
        CASE B - figure 9a-f

        densitymap() plots the density of households through the use of a gaussian filter
        over incrementalised count for varying levels of households wealth.
        """
        self.tit_cnt = 0
        titles = ['a','b','c','d','e','f']

        yrs = np.array([0, 10, 25])*4
        yrs = yrs[yrs<=sim.t*sim.t_inc]
        types = [1,2,3]     # proxy list for number of categories of wealth, ignore middle band

        for yr in yrs:
            yr = int(yr)
            cmaps = ['Greys', 'Reds', 'Greens']
            fig, axs = plt.subplots(1, 2)
            z = 0
            leg = ['$s_0$ < 20%', '80% < $s_0$']

            for x in range(len(types)):
                if x == 0 or x == 2:
                    pop_p = np.array(sim.all_store_pop[yr],dtype=float)
                    pop_p = np.array(sorted(pop_p, key=lambda x:x[2][0]))

                    # sim.p = np.array(sorted(sim.p, key=lambda x:x[2][0]))
                    
                    dwell_rcoord = np.array(sim.all_store_dwell[yr][:,1,[0,1]], dtype=float)
                    # X_p, Y_p = dwell_rcoord[:,0:2][np.array(pop_p[:,2],dtype=int)].T    # coordinates of each household
                    

                    if x == 0:
                        loc = dwell_rcoord[:,0:2][np.array(pop_p[:,0,2][:int((0.2*len(pop_p))+1)],dtype=int)]
                        # loc = sim.r[:,1,0:2][np.array(sim.p[:,0,2][:int((0.2*len(sim.p))+1)], dtype=int)]
                    elif x == 2:
                        loc = dwell_rcoord[:,0:2][np.array(pop_p[:,0,2][int((0.8*len(pop_p))+1):],dtype=int)]
                        # loc = sim.r[:,1,0:2][np.array(sim.p[:,0,2][int(0.8*len(sim.p)):], dtype=int)]

                    ## GAUSSIAN PLOT
                    # determine the density of households as by 0.25km increments
                    inc = 0.25
                    dim = int(int(sim.R//inc + 1))
                    locs = np.zeros(shape=(dim,dim))
                    X_p = loc[:,0]
                    Y_p = loc[:,1]

                    for x, y in zip(X_p, Y_p):
                        locs[int(x*4)][int(y*4)] += 1
                    sig = 3
                    locs = gaussian_filter(locs, sigma=sig)
                    C = locs.T

                    plt.rcParams['mathtext.fontset'] = 'stix'
                    plt.rcParams['font.family'] = 'STIXGeneral'
                    img = axs[z].imshow(C, cmap='Greys', interpolation='nearest', vmin=0, vmax=0.23)

                    ## KDE PLOT
                    # sns.kdeplot(loc[:, [0,1]], cmap='Greys', shade=True, bw_adjust=.5, ax=axs[z])
                    axs[z].set_title("({})".format(titles[self.tit_cnt]), fontdict={'family':'times','weight': 'bold','size': 24},loc='right')
                    # axs[z].set_xlabel('x (km)',fontsize=16, fontname='times')
                    # axs[z].set_ylabel('y (km)',fontsize=16, fontname='times')
                    axs[z].set_xlabel("{} (km)".format(r'$x$'), fontsize=18, fontname='times')
                    axs[z].set_ylabel("{} (km)".format(r'$y$'), fontsize=18, fontname='times')
                    # axs[z].set_xlim([-1, sim.R+1])
                    # axs[z].set_ylim([-1, sim.R+1])
                    axs[z].set_xticks(ticks=np.linspace(0,dim,6))
                    axs[z].set_xticklabels(labels=np.linspace(0,sim.R,6, dtype=int), rotation=0)
                    axs[z].set_yticks(ticks=np.linspace(0,dim,6))
                    axs[z].set_yticklabels(labels=np.linspace(0,sim.R,6, dtype=int))

                    for tick in axs[z].get_xticklabels():
                        tick.set_fontname("times")
                        tick.set_fontsize(14)
                    for tick in axs[z].get_yticklabels():
                        tick.set_fontname("times")
                        tick.set_fontsize(14)
                    self.tit_cnt += 1
                    axs[z].set_aspect('equal', adjustable='box')
                    axs[z].invert_yaxis()

                    # plt.colorbar(img, ax=axs)
                    z +=1

            plt.tight_layout()
            plt.savefig(sim.save_loc + '/urbanfig9{}{}'.format(titles[self.tit_cnt-2],titles[self.tit_cnt-1])+'.png', dpi=400)

    def meanB_r_10(self):
        """
        CASE B - figure 10a

        Calculated using the method of sorting ALL values by radial distance from centre, then smoothing to reduce noise.
        Alternatively, can also take mean of values lying with an annuli with incrementally increasing
        radii up to diameter equaling the size of the system.
        """

        yrs = np.array([0,1,2,5,10,25])*4
        yrs = yrs[yrs<=sim.t*sim.t_inc]
        plt.figure()

        for yr in yrs:
            yr = int(yr)


            if yr == 0:
                self.yr_plot = [0]
                self.c= 0
                # occ = np.where(dwell.r[:,1]==1)[0]
            else:
                self.yr_plot.append(yr+1)
            dwell_r = np.array(sim.all_store_dwell[yr][:,0], dtype=float)
            dwell_rcoord = np.array(sim.all_store_dwell[yr][:,1,[0,1]], dtype=float)
            dwell_B = np.array([sim.all_store_dwell[yr][:,2][:,0]], dtype=float).T

            occ = np.where(dwell_r[:,1]==0)[0]

            centric = dwell.centric  # U0 centricity location 
            euclid_cents = np.linalg.norm(dwell_rcoord[occ].T - np.array([centric[0]]).T, axis=0)

            r, B = np.array(sorted(zip(euclid_cents, dwell_B[occ,0]), key=lambda x:x[0])).T
            d = np.ceil(np.max(r))


            B0 = []
            mr = sim.R/2
            B = B[r<=12.5]
            r = r[r<=12.5]
            m = int(np.ceil(4*mr))
            for xx in range(m):
                r1 = (mr)*((xx)/m)
                r2 = (mr)*((xx+1)/m)
                B0.append(np.mean(B[np.where((r>=r1) & (r<r2))]))
            B0 = np.array(B,dtype=float)

            if yr >0:
                y = 'Year: {}'.format((yr+1)//4)
            else:
                y = 'Year: 0'
            # plt.rcParams['mathtext.fontset'] = 'stix'
            # plt.rcParams['font.family'] = 'STIXGeneral'
            colors = ['crimson','dodgerblue','forestgreen','orange','pink','black']
            plt.plot(r[::1],gaussian_filter(B0, sigma=100)/dwell.price_scale, label='{}'.format(y),color=colors[self.c])
            
            self.c+=1
            plt.legend(fontsize=14)
            t = r"$\langle B_0(r) \rangle \>\> (\$ \> 10^3 \> y^{-1})$"
            plt.ylabel("{}".format(t),fontsize=18)
            plt.xlabel("{} (km)".format(r'$r$'), fontsize=18)
            plt.xticks(ticks = np.linspace(0,12.5,6),labels=np.linspace(0,12.5,6),fontsize=14)
            plt.xlim([0,12.5])
            plt.title("(a)", fontdict={'family':'times','weight': 'bold','size': 24},loc='right')
        plt.savefig(sim.save_loc + '/urbanfig10a.png', dpi=400)

    def U_contribution(self):
        """
        CASE B - figure 10b,c

        Calculated using the method of sorting ALL values by radial distance from centre, then smoothing to reduce noise.
        Alternatively, can also take mean of values lying with an annuli with incrementally increasing
        radii up to diameter equaling the size of the system.
        """

        occ = pop.p[:,2]
        u = np.round(np.array(calcs.u.T, dtype=int), 2)
        s0 = np.round(np.array(calcs.s0, dtype=float),3)
        b0 = np.round(np.array(calcs.b0, dtype=float),3)

        m = calcs.m

        sb = np.round(np.array(calcs.sb, dtype=int), 2)
        u0 = np.round(np.array(calcs.u0, dtype=int), 2)
        qq = np.round(np.array(calcs.qq, dtype=int), 2)        

        centric = np.array([[0.5, 0.5]]) * dwell.R  # U0 centricity location 
        euclid_cents = np.linalg.norm(dwell.r_coord[occ].T - np.array([centric[0]]).T, axis=0)

        r, u, m,sb, u0, qq,s0,b0 = np.array(sorted(zip(euclid_cents, u[0],m, sb, u0, qq,s0,b0), key=lambda x:x[0])).T

        u_r = []
        u0_r = []
        qq_r = []
        sb_r = []
        s0_r = []
        b0_r = []
        mr = sim.R/2 #+ 0.1

        u = u[r<=12.5]
        m = m[r<=12.5]
        sb = sb[r<=12.5]
        u0 = u0[r<=12.5]
        qq = qq[r<=12.5]
        s0 = s0[r<=12.5]
        b0 = b0[r<=12.5]
        r = r[r<=12.5]


        m = int(np.ceil(4*mr))
        for xx in range(m):
            r1 = (mr)*((xx)/m)
            r2 = (mr)*((xx+1)/m)
            u_r.append(np.mean(u[np.where((r>=r1) & (r<r2))]))
            u0_r.append(np.mean(u0[np.where((r>=r1) & (r<r2))]))
            qq_r.append(np.mean(qq[np.where((r>=r1) & (r<r2))]))
            sb_r.append(np.mean(sb[np.where((r>=r1) & (r<r2))]))
            s0_r.append(np.mean(s0[np.where((r>=r1) & (r<r2))]))
            b0_r.append(np.mean(b0[np.where((r>=r1) & (r<r2))]))

        fig, ax = plt.subplots(1)
        y = 'contribution'
        y = r'$U(r) {}$'.format(y)
        colors = ['crimson','dodgerblue','forestgreen','orange','black']

        sb = np.array(sb,dtype=float)
        qq = np.array(qq,dtype=float)
        u = np.array(u, dtype=float)
        u0 = np.array(u0, dtype=float)
        print('Fig10b values:')
        print('q0q0 max: {}'.format(np.max(gaussian_filter(qq, sigma=50))))
        print('q0q0 min: {}'.format(np.min(gaussian_filter(qq, sigma=50))))
        print('------')
        print('s0B0 max: {}'.format(np.max(gaussian_filter(sb, sigma=50))))
        print('s0B0 min: {}'.format(np.min(gaussian_filter(sb, sigma=50))))
        print('------')
        print('U max: {}'.format(np.max(gaussian_filter(u, sigma=50))))
        print('U min: {}'.format(np.min(gaussian_filter(u, sigma=50))))
        # exit()

        l1 = ax.plot(r[::1], gaussian_filter(u, sigma=50)/dwell.price_scale, label=r'$U(r)$', color=colors[0])
        l3 = ax.plot(r[::1], gaussian_filter(sb, sigma=50)/dwell.price_scale, label=r'$(s_0\cdot B_0)(r)$', color=colors[2])
        l4 = ax.plot(r[::1], gaussian_filter(u0, sigma=10)/dwell.price_scale, label=r'$U_0(r)$', color=colors[3])
        l5 = ax.plot(r[::1], gaussian_filter(qq, sigma=50)/dwell.price_scale, label=r'$(q_0\cdot q_0)(r)$', color=colors[4])
        lns = l1 + l3 + l4 + l5
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, fontsize=16)

        d = np.ceil(np.max(r))

        t = r"$\langle U(r) \mathrm{ \>\> contribution} \rangle \>\> (\$ \> 10^3 \> y^{-1})$"
        ax.set_ylabel("{}".format(t),fontsize=18, fontname='times')
        ax.set_xlabel("{} (km)".format(r'$r$'), fontsize=18, fontname='times')
        ax.set_xlim([0,sim.R/2])
        plt.savefig(sim.save_loc + '/urbanfig10b.png', dpi=400)

        # USE FOR RAW VALUES
        ax.set_xticks(ticks = np.linspace(0,sim.R/2,6))
        ax.set_xticklabels(labels=np.linspace(0,sim.R/2,6))
        ax.set_xlim([0,sim.R/2])
        plt.title("(b)", fontdict={'family':'times','weight': 'bold','size': 24},loc='right')
        
        plt.figure()
        t1 = r"$\langle U(r) \rangle \>\> (\$ \> 10^3 \> y^{-1})$"
        plt.ylabel("{}".format(t1),fontsize=18, fontname='times')
        plt.plot(r[::1],gaussian_filter(u, sigma=50,mode='reflect')/dwell.price_scale, label=r'$U(r)$', color=colors[0])
        
        plt.xlabel("{} (km)".format(r'$r$'), fontsize=18, fontname='times')
        plt.xticks(ticks = np.linspace(0,sim.R/2,6),labels=np.linspace(0,sim.R/2,6))
        plt.xlim([0,sim.R/2])
        plt.ylim([0, 2.5])
        plt.title("(c)", fontdict={'family':'times','weight': 'bold','size': 24},loc='right')
        plt.savefig(sim.save_loc + '/urbanfig10c.png', dpi=400)

    def B1_dist(self):
        """
        CASE C - figure 11a

        B1_dist() plots spatial distribution of B1 characteristic in dwellings, depicted as heatmap.
        """

        C = dwell.B1_dist_val
        plt.figure()
        ax = sns.heatmap(C.T/dwell.price_scale, cmap='coolwarm')
        print(dwell.incs)
        plt.xticks(ticks=np.linspace(0,dwell.incs,6), labels=np.linspace(0,dwell.R,6,dtype=int),fontname='times',fontsize=14)
        plt.yticks(ticks=np.linspace(0,dwell.incs,6), labels=np.linspace(0,dwell.R,6,dtype=int),fontname='times',fontsize=14,rotation=0)
        plt.xticks(rotation=0)
        plt.xlabel('{} (km)'.format(r"$x$"), fontsize=18, fontname='times')
        plt.ylabel('{} (km)'.format(r"$y$"), fontsize=18, fontname='times')

        cb = ax.collections[0].colorbar
        b1 = r'$B_1(\bf{r})$'
        b2 = r'$\$ \> 10^3 \> y^{-1}$'
        cb.set_label(label='{} strength ({})'.format(b1,b2),fontsize=18)
        cb.set_ticks(np.linspace(-4,4,9))
        cb.set_ticklabels((np.linspace(-4,4,9,dtype=int)))

        cb.ax.tick_params(labelsize=14)

        plt.gca().invert_yaxis()
        plt.title("(a)", fontdict={'family':'times','weight': 'bold','size': 24},loc='right')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.savefig(sim.save_loc + '/urbanfig11a', dpi=400)

    def C_gaussian_map(self):
        """
        CASE C - figure 11b,c

        C_heatmap() creates two separate side-by-side plots of the gaussian density of household preference
        with s_1<0 and s_1>0 in blue and red respectively.
        """

        cmaps = ['Blues', 'Reds']
        fig, axs = plt.subplots(1,2)
        sloc = len(pop.s[0])-1
        for x in range(2):
            if x == 0:
                loc = sim.r[:,1,0:2][np.array(sim.p[:,0,2][np.where(sim.p[:,2,sloc]<0)], dtype=int)]
            elif x == 1:
                loc = sim.r[:,1,0:2][np.array(sim.p[:,0,2][np.where(sim.p[:,2,sloc]>0)], dtype=int)]
            X_p = loc[:,0]
            Y_p = loc[:,1]

            ## GAUSSIAN PLOT
            inc = 0.25
            dim = int(int(sim.R//inc + 1))#*(1//inc))
            locs = np.zeros(shape=(dim,dim))
            for xx, y in zip(X_p, Y_p):
                locs[int(xx*4)][int(y*4)] += 1

            sig = 3
            locs1 = gaussian_filter(locs, sigma=sig)

            # locs = locs-np.min(locs)
            # locs = locs/np.max(locs)
            C = locs1.T
            xx = 1-x
            axs[xx].imshow(C, cmap=cmaps[x], interpolation='nearest')
            axs[xx].invert_yaxis()
            plt.rcParams['mathtext.fontset'] = 'stix'
            plt.rcParams['font.family'] = 'STIXGeneral'

            axs[xx].set_xlabel('{} (km)'.format(r"$x$"), fontsize=18, fontname='times')
            axs[xx].set_ylabel('{} (km)'.format(r"$y$"), fontsize=18, fontname='times')

            axs[xx].set_xticks(ticks=np.linspace(0,sim.R*4+1,6, dtype=int))
            axs[xx].set_xticklabels(labels=np.linspace(0,sim.R,6, dtype=int), rotation=0)
            axs[xx].set_yticks(ticks=np.linspace(0,sim.R*4+1,6, dtype=int))
            axs[xx].set_yticklabels(labels=np.linspace(0,sim.R,6, dtype=int))
            axs[xx].set_aspect('equal', adjustable='box')

            for tick in axs[xx].get_xticklabels():
                tick.set_fontname("times")
                tick.set_fontsize(14)
            for tick in axs[xx].get_yticklabels():
                tick.set_fontname("times")
                tick.set_fontsize(14)
            title = ['b','c']
            axs[xx].set_title("({})".format(title[xx]), fontdict={'family':'times','weight': 'bold','size': 24},loc='right')

        plt.tight_layout()
        plt.savefig(sim.save_loc + '/urbanfig11bc.png', dpi=400)

    def C_heatmap(self):
        """
        CASE C 

        C_heatmap() plots distributon of household s_1 preference as a heatmap in one single image, ranging from
        -1 to +1.

        !!! NOT USED IN PAPER !!!
        """

        cmaps = ['Greys', 'Greys', 'Greens']
        
        sim_p = np.array(sorted(sim.p, key=lambda x:x[2][0]))        
        sloc = len(pop.s[0])-1
        s1_prefs = sim.p[:,2,sloc]
        X_p, Y_p = sim.r[:,1,0:2][np.array(sim.p[:,0,2], dtype=int)].T   # coordinates of each household
        
        scl = sim.R/10
        dim = sim.R
        inc = 1
        sz = dwell.R*inc + 1
        locs = np.zeros(shape=(sz,sz))
        for x, y, z in zip(X_p, Y_p, s1_prefs):
            locs[int(np.round(x)*inc)][int(np.round(y)*inc)] += z

        sig = 1
        locs = gaussian_filter(locs, sigma=sig)

        locs = (2*((locs-np.min(locs))/(np.max(locs)- np.min(locs)))-1)

        C = locs.T
        plt.figure(tight_layout=True)
        sns.heatmap(C, cmap='Greys')

        plt.xlabel('x (km)', fontsize=18, fontname='times')
        plt.ylabel('y (km)', fontsize=18, fontname='times')
        plt.gca().invert_yaxis()

        plt.title("(c)", fontdict={'family':'times','weight': 'bold','size': 24},loc='right')
        plt.xticks(ticks=np.linspace(0,sim.R,6),labels=np.linspace(0,sim.R,6,dtype=int),fontsize=14)
        plt.yticks(ticks=np.linspace(0,sim.R,6),labels=np.linspace(0,sim.R,6,dtype=int),fontsize=14, rotation=0)
        plt.savefig(sim.save_loc + '/C_heatmap-'+ sim.sim_id +'.png', dpi=400)

    def spatial_types(self):
        """ 
        CASE D - figure 12(all), 13

        Spatial plot of households highlighting characteristic type q_1(p).
        """

        X_r, Y_r = sim.r[:,1][:,0:2].T

        # sort by characteristic type
        self.q_loc = len(pop.q[0])-1
        sim.p = np.array(sorted(sim.p, key=lambda x:x[1][self.q_loc]))
        types = [sum([1 for x in sim.p[:,1,self.q_loc] if x == pop.q_bound['type I/II'][y]]) for y in range(len(pop.q_bound['type I/II'])) ]
        self.types = types

        colours = ['midnightblue','crimson','orange','forestgreen','darkorange']

        self.edge = ['lightgrey']*len(X_r)
        self.c = [0]*len(X_r)

        for y, x in enumerate(types):
            self.edge.extend([colours[y]]*x)
            self.c.extend([y+1]*x)

        cmap = ['lightgrey'] + colours[:len(types)]
        leg = ['Dwellings'] + ['Type {}'.format(x) for x in ['-1','+1']]
        self.cmap = ListedColormap(cmap)

        X_p, Y_p = sim.r[:,1,0:2][np.array(sim.p[:,0,2], dtype=int)].T   # coordinates of each household
        self.X_c = np.concatenate((X_r, X_p), axis=0)
        self.Y_c = np.concatenate((Y_r, Y_p), axis=0)
        self.s = [10]*len(X_r) + [10]*len(X_p)
        self.mark = [str(".")]*len(X_r) + [str(".")]*len(X_p)

        fig, ax = plt.subplots(1)
        sc = plt.scatter(self.X_c, self.Y_c, s=self.s, c=self.c, edgecolors=self.edge, cmap=self.cmap, marker='.')

        plt.xlabel('{} (km)'.format(r'$x$'), fontsize=18, fontname='times')
        plt.ylabel('{} (km)'.format(r'$y$'), fontsize=18, fontname='times')
        plt.gca().set_aspect('equal', adjustable='box')

        lims = 0.02*sim.R
        plt.xlim([-lims, sim.R+lims])
        plt.ylim([-lims, sim.R+lims])
        plt.xticks(ticks=np.linspace(0,sim.R,6),labels=np.linspace(0,sim.R,6,dtype=int))
        plt.yticks(ticks=np.linspace(0,sim.R,6),labels=np.linspace(0,sim.R,6,dtype=int))

        if sim.full_case == 'D':
            plt.title("(a)", fontdict={'family':'times','weight': 'bold','size': 24},loc='right')
            plt.tight_layout()
            plt.savefig(sim.save_loc + '/urbanfig12a.png', dpi=400)
        else:
            plt.tight_layout()
            plt.savefig(sim.save_loc + '/urbanfig13.png', dpi=400)


    def spatial_types_movie(self):
        """ 
        CASE D - figure 12(.), 13

        Spatial plot of households highlighting characteristic type q_1(p).
        """
        yrs = np.arange(0,101,dtype=int)    # this is for movie creation -- save all figures to folder then use
        # https://ezgif.com/maker to compile pics into .gif
        yrs = yrs[yrs<=sim.t*sim.t_inc]
        for yr in yrs:
            yr = int(yr)

            dwell_r = np.array(sim.all_store_dwell[yr][:,0], dtype=float)
            dwell_rcoord = np.array(sim.all_store_dwell[yr][:,1,[0,1]], dtype=float)
            X_r, Y_r = dwell_rcoord[:,0:2].T

            # sort by characteristic type
            pop_p = np.array(sim.all_store_pop[yr],dtype=float)

            self.q_loc = 1
            sim.p = np.array(sorted(pop_p, key=lambda x:x[1][self.q_loc]))
            types = [sum([1 for x in sim.p[:,1,self.q_loc] if x == pop.q_bound['type I/II'][y]]) for y in range(len(pop.q_bound['type I/II'])) ]
            self.types = types

            colours = ['midnightblue','crimson','orange','forestgreen','darkorange']

            self.edge = ['lightgrey']*len(X_r)
            self.c = [0]*len(X_r)

            for y, x in enumerate(types):
                self.edge.extend([colours[y]]*x)
                self.c.extend([y+1]*x)

            cmap = ['lightgrey'] + colours[:len(types)]
            leg = ['Dwellings'] + ['Type {}'.format(x) for x in ['-1','+1']]
            self.cmap = ListedColormap(cmap)

            X_p, Y_p = dwell_rcoord[:,0:2][np.array(sim.p[:,0,2], dtype=int)].T   # coordinates of each household
            self.X_c = np.concatenate((X_r, X_p), axis=0)
            self.Y_c = np.concatenate((Y_r, Y_p), axis=0)
            self.s = [10]*len(X_r) + [10]*len(X_p)
            self.mark = [str(".")]*len(X_r) + [str(".")]*len(X_p)

            fig, ax = plt.subplots(1)
            sc = plt.scatter(self.X_c, self.Y_c, s=self.s, c=self.c, edgecolors=self.edge, cmap=self.cmap, marker='.')

            plt.xlabel('{} (km)'.format(r'$x$'), fontsize=18, fontname='times')
            plt.ylabel('{} (km)'.format(r'$y$'), fontsize=18, fontname='times')
            plt.gca().set_aspect('equal', adjustable='box')

            lims = 0.02*sim.R
            plt.xlim([-lims, sim.R+lims])
            plt.ylim([-lims, sim.R+lims])
            plt.xticks(ticks=np.linspace(0,sim.R,6),labels=np.linspace(0,sim.R,6,dtype=int))
            plt.yticks(ticks=np.linspace(0,sim.R,6),labels=np.linspace(0,sim.R,6,dtype=int))

            # plt.title("(a)", fontdict={'family':'times','weight': 'bold','size': 24},loc='right')
            plt.tight_layout()
            plt.savefig(sim.save_loc + '/D_spatial-types_50_{}-'.format(yr)+ sim.sim_id +'.png', dpi=400)
            plt.close()

    def gaussian_types(self):
        """
        CASE D

        gaussian_types() plots gaussian density OR KDE plot of the household density 
        as according to household characteristic type q1.
        """

        fig, axs = plt.subplots(1,len(self.types))
        cmaps=['Blues','Reds','Greens']

        for x in range(len(self.types)):
            loc = sim.r[:,1,0:2][np.array(sim.p[:,0,2][np.where(sim.p[:,1,self.q_loc]== pop.q_bound['type I/II'][x])], dtype=int)]
            X_p = loc[:,0]
            Y_p = loc[:,1]

            ## GAUSSIAN PLOT
            inc = 0.25
            dim = int(int(sim.R//inc + 1))
            print(dim)
            locs = np.zeros(shape=(dim,dim))
            for xx, y in zip(X_p, Y_p):
                locs[int(xx*4)][int(y*4)] += 1

            sig = 3
            locs1 = gaussian_filter(locs, sigma=sig)

            C = locs1.T
            axs[x].imshow(C, cmap=cmaps[x], interpolation='nearest')
            axs[x].invert_yaxis()

            ## KDE PLOT
            # sns.kdeplot(loc[:,[0,1]], cmap='Greys', shade=True, bw=2, ax=axs[x])
            axs[x].set_xlabel('{} (km)'.format(r'$x$'),fontsize=16, fontname='times')
            axs[x].set_ylabel('{} (km)'.format(r'$y$'),fontsize=16, fontname='times')

            axs[x].set_xticks(ticks=np.linspace(0,sim.R*4+1,6, dtype=int))
            axs[x].set_xticklabels(labels=np.linspace(0,sim.R,6, dtype=int), rotation=0)
            axs[x].set_yticks(ticks=np.linspace(0,sim.R*4+1,6, dtype=int))
            axs[x].set_yticklabels(labels=np.linspace(0,sim.R,6, dtype=int))
            axs[x].set_aspect('equal', adjustable='box')
            axs[x].set_title('Type {} household'.format(pop.q_bound['type I/II'][x]))
        plt.tight_layout()
        plt.savefig(sim.save_loc + '/D_gaussian_types-'+ sim.sim_id +'.png', dpi=400)

    def M_t(self):
        """
        CASE E

        M_t() plots total number of dwellings M vs time t.
        !!! NOT USED IN PAPER !!!
        """

        plt.figure()
        M = []
        for x in sim.all_store_dwell:
            M.append(np.max(x[:,0,0]))

        plt.plot(M)
        plt.xticks(ticks=np.linspace(0,sim.t*sim.t_inc,6), labels=np.linspace(0,sim.t,6, dtype=int))
        plt.xlabel('t (years)')
        plt.ylabel('Dwellings M')
        plt.savefig(sim.save_loc + '/E_M_t-'+ sim.sim_id +'.png', dpi=400)
        
    def N_t(self):
        """
        CASE E - figure 14a

        N_t() plots total number of households N vs time t.
        """

        plt.figure()
        N = []
        for x in sim.all_store_pop:
            N.append(np.max(x[:,0,0]))

        plt.plot(N,color='black')
        plt.xticks(ticks=np.linspace(0,sim.t*sim.t_inc,6), labels=np.linspace(0,sim.t,6, dtype=int),fontsize=16)
        plt.xlabel('t (years)', fontsize=20, fontname='times')
        y = r'N'
        plt.yticks(fontsize=16)
        plt.ylabel('Households {}'.format(y), fontsize=20, fontname='times')
        plt.xlim([0, len(N)])
        plt.ylim([0, np.max(N)+500])
        plt.title("(a)", fontdict={'family':'times','weight': 'bold','size': 24},loc='right')
        plt.savefig(sim.save_loc + '/urbanfig14a.png', dpi=400,bbox_inches="tight")

    def meanB_t_r(self):
        """
        CASE E - figure 14b

        meanB_t_r() plots mean dwelling price as a function of time at radii r from list of distances from centre.

        calculated using price of all dwellings lying within annulus of width 1 (+-0.5) at specific radii
        """

        plt.figure()
        yrs = np.linspace(0,len(sim.all_store_dwell), len(sim.all_store_dwell)+1,dtype=int)
        yrs = yrs[yrs<=sim.t*sim.t_inc]

        for yr in yrs:
            yr = int(yr)

            if yr<sim.t*sim.t_inc:
                dwell_r = np.array(sim.all_store_dwell[yr][:,0], dtype=float)
                dwell_rcoord = np.array(sim.all_store_dwell[yr][:,1,[0,1]], dtype=float)
                dwell_B = np.array([sim.all_store_dwell[yr][:,2][:,0]], dtype=float).T

                occ = np.where(dwell_r[:,1]==0)[0]
                
                centric = dwell.centric  # U0 centricity location 
                euclid_cents = np.linalg.norm(dwell_rcoord[occ].T - np.array([centric[0]]).T, axis=0)

                r, B = np.array(sorted(zip(euclid_cents, dwell_B[occ,0]), key=lambda x:x[0])).T
                d = np.max(r)

                self.rads = [1,3,5,7,10,12.5]
                tmp_r = []

                for x in self.rads:
                    tmp_r.append(np.mean(B[np.where((r>x-0.5)&(r<x+0.5))]))

                self.rad_t.append(tmp_r)
            else:
                self.rad_t = np.array(self.rad_t).T

                plt.rcParams['mathtext.fontset'] = 'stix'
                plt.rcParams['font.family'] = 'STIXGeneral'
                colors = ['crimson','dodgerblue','forestgreen','orange','black','pink','midnightblue']

                for y,x in enumerate(self.rad_t):
                    plt.plot(gaussian_filter(x,sigma=4)/dwell.price_scale, label='R = {} km'.format(self.rads[y]), color=colors[y])
                
                plt.legend(fontsize=14,loc='upper right')
                plt.xlabel("t (years)", fontsize=20, fontname='times')
                plt.xticks(ticks=np.linspace(0,sim.t*sim.t_inc,6),labels=np.linspace(0,sim.t,6,dtype=int),fontsize=16)
                plt.yticks(fontsize=16)
                t = r"$\langle B_0(r) \rangle \>\> (\$ \> 10^3 \> y^{-1})$"
                plt.ylabel("{}".format(t),fontsize=20)
                m_min = np.min(np.min(self.rad_t))
                m_max = np.max(np.max(self.rad_t))

                if hasattr(sim, 'new_min')==True:
                    plt.plot([sim.new_min, sim.new_min], [0, 10], c='black')
                if hasattr(sim, 'new_max')==True:
                    plt.plot([sim.new_max, sim.new_max], [0, 10], c='black')
                plt.xlim([0, np.size(gaussian_filter(x,sigma=4)/dwell.price_scale)])
                plt.ylim([0, 10])
                plt.title("(b)", fontdict={'family':'times','weight': 'bold','size': 24},loc='right')
        plt.savefig(sim.save_loc + '/urbanfig14b.png', dpi=400,bbox_inches="tight")

    def meanB_r_t(self):
        """
        CASE E - figure 14c

        meanB_r_t() plots mean dwelling price as a function of radius at time points t from a list of years.
        
        Calculated using price of all dwellings lying within annulus with incrementally increasing
        radii up to the R2 = system size R/2.
        """

        max_dist = 0

        plt.figure()
        yrs = np.array([0,19.75,39.75,59.75,69.75,79.75])*4
        yrs = np.array([19.75,39.75,59.75,79.75])*4
        yrs = yrs[yrs<=sim.t*sim.t_inc]

        for t,yr in enumerate(yrs):
            yr = int(yr)

            if t == 0:
                self.yr_plot = [0]
                self.c= 0
                self.nmax = 0
                # occ = np.where(dwell.r[:,1]==1)[0]    # get ALL on-market dwellings
            else:
                self.yr_plot.append(yr+1)

            """ get data from store """

            dwell_r = np.array(sim.all_store_dwell[yr][:,0], dtype=float)
            dwell_rcoord = np.array(sim.all_store_dwell[yr][:,1,[0,1]], dtype=float)
            dwell_B = np.array([sim.all_store_dwell[yr][:,2][:,0]], dtype=float).T
            occ = np.where(dwell_r[:,1]==0)[0]

            centric = dwell.centric  # U0 centricity location 
            euclid_cents = np.linalg.norm(dwell_rcoord[occ].T - np.array([centric[0]]).T, axis=0)

            r, B = np.array(sorted(zip(euclid_cents, dwell_B[occ,0]), key=lambda x:x[0])).T

            """ plotting """
            if yr >0:
                y = 'Year: {}'.format((yr+1)//4)
            else:
                y = 'Year: 0'
            plt.rcParams['mathtext.fontset'] = 'stix'
            plt.rcParams['font.family'] = 'STIXGeneral'
            colors = ['crimson','dodgerblue','forestgreen','orange','black','pink','midnightblue']
            
            B0 = []
            # r = r[r<=12.5]
            if max_dist == 1:

                mr = ((sim.R/2)**2 + (sim.R/2)**2)**0.5
            else:
                mr = sim.R/2
                B = B[r<=12.5]

            d = np.ceil(4*mr)
            m = int(d)
            for xx in range(m):
                r1 = (mr)*((xx)/m)
                r2 = (mr)*((xx+1)/m)
                B0.append(np.mean(B[np.where((r>=r1) & (r<r2))]))

            if yr == 0:
                B0 = B0[1:]
                r_b = np.linspace(1,len(B0),len(B0))
                plt.plot(r_b,gaussian_filter(B0, sigma=2)/dwell.price_scale, label='{}'.format(y),color=colors[self.c])
            else:
                r_b = np.linspace(1,len(B0),len(B0))
                B0 = np.array(B0)
                if np.isnan(sum(B0)) == True:
                    nans, x= np.isnan(B0), lambda z: z.nonzero()[0]
                    B0[nans] = np.interp(x(nans), x(~nans), B0[~nans])
                plt.plot(r_b,gaussian_filter(B0, sigma=2)/dwell.price_scale, label='{}'.format(y),color=colors[self.c])

            self.c+=1
            plt.legend(fontsize=14, loc='lower left')
            t = r"$\langle B_0(r) \rangle \>\> (\$ \> 10^3 \> y^{-1})$"
            plt.ylabel("{}".format(t),fontsize=20)
            plt.yticks(fontsize=16)
            plt.xticks(ticks = np.linspace(0,m-1,6),labels=np.linspace(0,mr,6 ),fontsize=16)
            plt.xlabel("{} (km)".format(r'$r$'), fontsize=20, fontname='times')
            plt.xlim([0, m-1])

            if (np.nanmax(gaussian_filter(B0, sigma=2))/dwell.price_scale)*1.25 > self.nmax:
                self.nmax = (np.nanmax(gaussian_filter(B0, sigma=2))/dwell.price_scale)*1.25
            plt.ylim([0, self.nmax])
            plt.title("(c)", fontdict={'family':'times','weight': 'bold','size': 24},loc='right')

        plt.savefig(sim.save_loc + '/urbanfig14c.png', dpi=400,bbox_inches="tight")

    def mean_density(self):
        """
        CASE E - figure 14d

        Density calculated using annuli with incrementally increasing radii.
        Area of annuli calculated up to R2 = system size R/2. This is not largest distance from
        centre; largest distance from centre = sqrt((sim.R/2)^2 + (sim.R/2)^2).

        Can reduce to R=12.5km (halfwidth of simulation) OR R=(R**2+R**2)**0.5 (max dist from centre)
        """

        max_dist = 0

        plt.figure()
        # yrs = np.array([0,19.75,39.75,59.75,69.75,79.75])*4
        yrs = np.array([19.75,39.75,59.75,79.75])*4
        yrs = yrs[yrs<=sim.t*sim.t_inc]

        for t,yr in enumerate(yrs):
            yr = int(yr)

            if t == 0:
                self.yr_plot = [0]
                self.c= 0
                self.nmax = 0
            else:
                self.yr_plot.append(yr+1)

            dwell_r = np.array(sim.all_store_dwell[yr][:,0], dtype=float)
            dwell_rcoord = np.array(sim.all_store_dwell[yr][:,1,[0,1]], dtype=float)
            dwell_B = np.array([sim.all_store_dwell[yr][:,2][:,0]], dtype=float).T

            occ = np.where(dwell_r[:,1]==0)[0]

            centric = dwell.centric  # U0 centricity location 
            euclid_cents = np.linalg.norm(dwell_rcoord[occ].T - np.array([centric[0]]).T, axis=0)

            r, B = np.array(sorted(zip(euclid_cents, dwell_B[occ,0]), key=lambda x:x[0])).T
            # r = r[r<=12.5]
            if max_dist == 1:
                mr = ((sim.R/2)**2 + (sim.R/2)**2)**0.5
            else:
                mr = sim.R/2
                B = B[r<=12.5]
            d = np.ceil(4*mr)
            m = int(d)
            cnt = np.zeros(shape=(int(d),1))
            sz = np.zeros(shape=(m,1))

            # these next two lines and the lines way below in the radius loop are for 
            # plotting circles of the area calculation.
            # plt.figure()
            # plt.scatter(dwell.r_coord[occ,0],dwell.r_coord[occ,1],alpha=1, s=10, marker='.')
            for xx in range(m):

                r2 = (mr)*((xx+1)/m)
                c2 = np.pi*(r2)**2
                r1 = (mr)*((xx)/m)
                c1 = np.pi*(r1)**2
                R = 12.5

                cnt[xx] += np.size(np.where((r>=r1) & (r<r2)))

                if max_dist == 0:
                    if r2 > R and r1 <R:
                        s2 = 0.5*r2**2 * (2*np.arccos(R/r2) - np.sin(2*np.arccos(R/r2)))
                        A1 = c2 - c1
                        A2 = s2 
                        A = A1 - 4*A2

                    elif r2>=R and r1>=R:
                        s2 = 0.5*r2**2 * (2*np.arccos(R/r2) - np.sin(2*np.arccos(R/r2)))
                        s1 = 0.5*r1**2 * (2*np.arccos(R/r1) - np.sin(2*np.arccos(R/r1)))
                        A1 = c2 - c1
                        A2 = s2 - s1
                        A = A1 - 4*A2

                    else:
                        A = c2 - c1
                elif max_dist == 1:
                    if r2 > R and r1 <R:
                        s2 = 0.5*r2**2 * (2*np.arccos(R/r2) - np.sin(2*np.arccos(R/r2)))
                        A1 = c2 - c1
                        A2 = s2 
                        A = A1 - 4*A2

                    elif r2>=R and r1>=R:
                        s2 = 0.5*r2**2 * (2*np.arccos(R/r2) - np.sin(2*np.arccos(R/r2)))
                        s1 = 0.5*r1**2 * (2*np.arccos(R/r1) - np.sin(2*np.arccos(R/r1)))
                        A1 = c2 - c1
                        A2 = s2 - s1
                        A = A1 - 4*A2
                    
                    elif r2 >= mr and r1<mr:
                        s1 = 0.5*r1**2 * (2*np.arccos(R/r1) - np.sin(2*np.arccos(R/r1)))
                        c1 = c1 - 4*s1
                        A = (2*R)**2 - c1

                    else:
                        A = c2 - c1
                sz[int(xx)] += A

                # plot circles
                # circle = plt.Circle((sim.R/2, sim.R/2), r2, color='r', fill=False)
                # plt.xlim([-0.5,sim.R+0.5])
                # plt.ylim([-0.5,sim.R+0.5])
                # ax = plt.gca()
                # ax.add_patch(circle)

            # average density
            rho = cnt/sz

            if yr >0:
                y = 'Year: {}'.format((yr+1)//4)
            else:
                y = 'Year: 0'
            plt.rcParams['mathtext.fontset'] = 'stix'
            plt.rcParams['font.family'] = 'STIXGeneral'

            colors = ['crimson','dodgerblue','forestgreen','orange','pink','black','midnightblue']

            plt.plot(gaussian_filter(rho, sigma=2), label='{}'.format(y),color=colors[self.c])
            # plt.plot(rho, label='{}'.format(y),color=colors[self.c])
            # plt.plot(cnt, label='household count')
            # plt.plot(sz, label='area')
            self.c+=1

            plt.legend(fontsize=14,loc='lower left')
            t = r"$d(r) \>\> (N\>$".format('km')
            t1 = r"$^{-2})$"
            plt.ylabel("{} km{}".format(t,t1),fontsize=20, fontname='times')
            plt.yticks(fontsize=16,fontname='times')

            plt.xticks(ticks = np.linspace(0,d-1,6),labels=np.round(np.linspace(0,mr,6),1),fontsize=16)
            plt.xlabel("{} (km)".format(r'$r$'), fontsize=20, fontname='times')
            plt.xlim([0,d-1])
            # plt.xlim([0,5])
            # plt.ylim([0, 50])

            if np.max(gaussian_filter(rho, sigma=2))*1.25 > self.nmax:
                self.nmax = np.max(gaussian_filter(rho, sigma=2))*1.25
            # plt.ylim([0, self.nmax])
            plt.title("(d)", fontdict={'family':'times','weight': 'bold','size': 24},loc='right')

        plt.savefig(sim.save_loc + '/urbanfig14d.png', dpi=400,bbox_inches="tight")

    def spatial_dist(self):
        """
        !! EXTRA VISUALISATION FUNCTION !!
        !!! NOT USED IN PAPER !!!

        spatial_dist() plots spatial distribution (1), gaussian density (2), KDE plot (3) of households;
        dwellings also in (1).
        """

        sim.step = 'END'

        # Ammended code from below:
        # https://stackoverflow.com/questions/7908636/possible-to-make-labels-appear-when-hovering-over-a-point-in-matplotlib 

        X_r, Y_r = sim.r[:,1][:,0:2].T      # coordinates of dwelling
        
        # if pop.p.relig == 1:
        # sim.p = np.array(sorted(sim.p, key=lambda x:x[0][3]))
        sim.p = np.array(sorted(sim.p, key=lambda x:x[2][0]))   # sort by wealth

        types = [sum([1 for x in sim.p[:,1,2] if x == pop.q_bound['type I/II'][y]]) for y in range(len(pop.q_bound['type I/II'])) ]
        types = [len(sim.p)]
        self.types = types

        # type_I = sum([1 for x in sim.p[:,1,2] if x == pop.q_bound['type I/II'][0]])
        # type_II = sum([1 for x in sim.p[:,1,2] if x == pop.q_bound['type I/II'][1]])
        # type_III = sum([1 for x in sim.p[:,1,2] if x == pop.q_bound['type I/II'][2]])
        # type_IV = sum([1 for x in sim.p[:,1,2] if x == pop.q_bound['type I/II'][3]])
        colours = ['Black','crimson','forestgreen'] #'darkorange'
        # colours = ['C0', 'C1', 'C2', 'C3', 'C4', 'C6', 'C8', 'C9']

        self.edge = ['lightgrey']*len(X_r)
        self.c = [0]*len(X_r)

        for y, x in enumerate(types):
            self.edge.extend([colours[y]]*x)
            self.c.extend([y+1]*x)

        cmap = ['lightgrey'] + colours[:len(types)]
        leg = ['Dwellings', 'Households']# + ['Type {}'.format(x+1) for x in range(len(types))]
        self.cmap = ListedColormap(cmap)

        # self.edge = ['C7']*len(X_r) + ['C0']*type_I + ['C1'] *type_II + ['C3']*type_III + ['C2']*type_IV
        # self.c = [0]*len(X_r) + [1]*type_I + [2]*type_II + [3]*type_III + [4]*type_IV
        # self.cmap = ListedColormap(['lightgrey', 'C0', 'C1', 'C2', 'C3'])
        # leg = ['Dwellings','Type I', 'Type II', 'Type III', 'Type IV']

        X_p, Y_p = sim.r[:,1,0:2][np.array(sim.p[:,0,2], dtype=int)].T   # coordinates of each household
        self.X_p, self.Y_p = X_p, Y_p
        
        self.loc = sim.r[:,1][np.array(sim.p[:,0,2], dtype=int)]

        # self.r_names = ["U0: {} \n r: ({}, {}) \n B: {} \n -------- \n".format(round(x[0][2],1), x[1][0],x[1][1], [round(y,1) for y in x[2][:3]]) for x in sim.r]
        # self.p_names = ["p: {}, U: {}, n_T: {}, r_ID: {} \n q: {} \n s: {}\n G: {}".format(x[0][0], round(x[0][4][0],1), x[0][3], x[0][2], [round(y,3) for y in x[1]], \
                        # [round(y,3) for y in x[2][:3]], [round(y,3) for y in x[3][:3]]) for x in sim.p]
        
        # household of interest
        # houseX, houseY = sim.r[:,1,0:2][np.array(sim.p[np.where(sim.p[:,0,0]==576),0,2],dtype=int)][0][0]
        houseX, houseY = sim.r[:,1,0:2][np.array(sim.p[-50:,0,2], dtype=int)].T

        # self.names = np.concatenate((self.r_names, self.p_names),axis=0)
        self.X_c = np.concatenate((X_r, X_p), axis=0)
        self.Y_c = np.concatenate((Y_r, Y_p), axis=0)
        self.s = [10]*len(X_r) + [10]*len(X_p)
        self.mark = [str(".")]*len(X_r) + [str(".")]*len(X_p)
        # self.edge = ['C7']*len(X_r) + ['C3']*len(X_p)
        # self.c = [0]*len(X_r) + [1]*len(X_p)
        # self.cmap = ListedColormap(['lightgrey', 'C3'])
        titles = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s']
        fig, axs = plt.subplots(1,3, figsize=(12, 4))
        ax = 0
        axs[ax].scatter(self.X_c, self.Y_c, s=self.s, c=self.c, edgecolors=self.edge, cmap=self.cmap, marker='.')
        # axs[0].scatter(houseX, houseY, s=[50]*len(houseX), c=[1]*len(houseX), edgecolors='crimson', cmap=ListedColormap(['crimson']), marker='.')

        axs[ax].set_xlabel('x (km)',fontsize=16, fontname='times')
        axs[ax].set_ylabel('y (km)',fontsize=16, fontname='times')
        axs[ax].set_xlim([-1, 26])
        axs[ax].set_ylim([-1, 26])
        axs[ax].set_xticks(ticks=np.linspace(0,sim.R,6, dtype=int))
        axs[ax].set_yticks(ticks=np.linspace(0,sim.R,6, dtype=int))
        for tick in axs[ax].get_xticklabels():
            tick.set_fontname("times")
            tick.set_fontsize(12)
        for tick in axs[ax].get_yticklabels():
            tick.set_fontname("times")
            tick.set_fontsize(12)
        axs[ax].set_aspect('equal', adjustable='box')
        axs[ax].set_title('spatial')
        lims = 0.02*sim.R
        axs[ax].set_xlim([-lims, sim.R+lims])
        axs[ax].set_ylim([-lims, sim.R+lims])
        # axs[0].set_title("({})".format(titles[self.tit_cnt]), fontdict={'family':'times','weight': 'bold','size': 20},loc='right')
        # self.tit_cnt += 1



        # determine the density of households as by 0.25 increments
        inc = 0.25
        dim = int(int(sim.R//inc + 1))#*(1//inc))
        locs = np.zeros(shape=(dim,dim))
        for x, y in zip(self.X_p, self.Y_p):
            locs[int(x*4)][int(y*4)] += 1
        
        ###### DONT NEED THIS BLOCK - ONLY FOR PLOTTING GAUSSIAN SMOOTHING######
        # sig = 1
        # locs1 = gaussian_filter(locs, sigma=sig)
        # # locs = locs-np.min(locs)
        # # locs = locs/np.max(locs)
        # C = locs1.T
        # axs[0].imshow(C, cmap='Blues', interpolation='nearest')
        # axs[0].set_xlabel('x (km)',fontsize=16, fontname='times')
        # axs[0].set_ylabel('y (km)',fontsize=16, fontname='times')
        # for tick in axs[0].get_xticklabels():
        #     tick.set_fontname("times")
        #     tick.set_fontsize(12)
        # for tick in axs[0].get_yticklabels():
        #     tick.set_fontname("times")
        #     tick.set_fontsize(12)
        # # axs[0].set_xlim([-1, sim.R+1])
        # # axs[0].set_ylim([-1, sim.R+1])
        # axs[0].set_xticks(ticks=np.linspace(0,sim.R*4+1,6, dtype=int))
        # axs[0].set_xticklabels(labels=np.linspace(0,sim.R,6, dtype=int), rotation=0)
        # axs[0].set_yticks(ticks=np.linspace(0,sim.R*4+1,6, dtype=int))
        # axs[0].set_yticklabels(labels=np.linspace(0,sim.R,6, dtype=int))
        # axs[0].set_aspect('equal', adjustable='box')
        # axs[0].set_title(r'$\sigma=1$')
        # axs[0].invert_yaxis()

        # sig = 2
        # locs1 = gaussian_filter(locs, sigma=sig)
        # # locs = locs-np.min(locs)
        # # locs = locs/np.max(locs)
        # C = locs1.T
        # axs[1].imshow(C, cmap='Blues', interpolation='nearest')
        # axs[1].set_xlabel('x (km)',fontsize=16, fontname='times')
        # axs[1].set_ylabel('y (km)',fontsize=16, fontname='times')
        # for tick in axs[1].get_xticklabels():
        #     tick.set_fontname("times")
        #     tick.set_fontsize(12)
        # for tick in axs[1].get_yticklabels():
        #     tick.set_fontname("times")
        #     tick.set_fontsize(12)
        # # axs[1].set_xlim([-1, sim.R+1])
        # # axs[1].set_ylim([-1, sim.R+1])
        # axs[1].set_xticks(ticks=np.linspace(0,sim.R*4+1,6, dtype=int))
        # axs[1].set_xticklabels(labels=np.linspace(0,sim.R,6, dtype=int), rotation=0)
        # axs[1].set_yticks(ticks=np.linspace(0,sim.R*4+1,6, dtype=int))
        # axs[1].set_yticklabels(labels=np.linspace(0,sim.R,6, dtype=int))
        # axs[1].set_aspect('equal', adjustable='box')
        # axs[1].set_title(r'$\sigma=2$')
        # axs[1].invert_yaxis()

        #######     ######       ######     ######      ######
        sig = 3
        ax += 1
        locs1 = gaussian_filter(locs, sigma=sig)

        # locs = locs-np.min(locs)
        # locs = locs/np.max(locs)
        C = locs1.T
        axs[ax].imshow(C, cmap='Blues', interpolation='nearest')
        axs[ax].set_xlabel('x (km)',fontsize=16, fontname='times')
        axs[ax].set_ylabel('y (km)',fontsize=16, fontname='times')
        for tick in axs[ax].get_xticklabels():
            tick.set_fontname("times")
            tick.set_fontsize(12)
        for tick in axs[ax].get_yticklabels():
            tick.set_fontname("times")
            tick.set_fontsize(12)
        # axs[2].set_xlim([-1, sim.R+1])
        # axs[2].set_ylim([-1, sim.R+1])
        axs[ax].set_xticks(ticks=np.linspace(0,sim.R*4+1,6, dtype=int))
        axs[ax].set_xticklabels(labels=np.linspace(0,sim.R,6, dtype=int), rotation=0)
        axs[ax].set_yticks(ticks=np.linspace(0,sim.R*4+1,6, dtype=int))
        axs[ax].set_yticklabels(labels=np.linspace(0,sim.R,6, dtype=int))
        axs[ax].set_aspect('equal', adjustable='box')
        axs[ax].set_title(r'$\sigma=3$')
        axs[ax].invert_yaxis()

        t = np.array(self.loc[:,[0,1]],dtype=np.float)
        t = t.flatten()
        kde = scipy.stats.gaussian_kde(t)
        
        # determining bandwidth from KDE plot function
        # kde.set_bandwidth(kde.factor * 1)
        # bw = np.sqrt(kde.covariance.squeeze())
        # bw = scipy.stats.gaussian_kde.covariance_factor(self.loc[:,[0,1]])
        # print(kde.covariance_factor()*np.std(self.loc[:,[0,1]]))
        # print(kde.neff)

        std = kde.covariance_factor()*np.std(self.loc[:,[0,1]])*1.2#*0.99
        adj = 2/std
        # print(std)

        ax += 1
        # exit()
        sns.kdeplot(self.loc[:,[0,1]], cmap='Blues', shade=True, bw=2, ax=axs[ax])
        # sns.kdeplot(test, cmap='Blues', shade=True, bw=1, ax=axs[1])
        # sns.kdeplot(self.loc[:,[0,1]], cmap='Blues', shade=True, bw_adjust=.5, ax=axs[1])
        axs[ax].set_xlabel('x (km)',fontsize=16, fontname='times')
        axs[ax].set_ylabel('y (km)',fontsize=16, fontname='times')
        for tick in axs[ax].get_xticklabels():
            tick.set_fontname("times")
            tick.set_fontsize(12)
        for tick in axs[ax].get_yticklabels():
            tick.set_fontname("times")
            tick.set_fontsize(12)

        lims = 0.02*sim.R
        axs[ax].set_xlim([-lims, sim.R+lims])
        axs[ax].set_ylim([-lims, sim.R+lims])
        axs[ax].set_xticks(ticks=np.linspace(0,sim.R,6, dtype=int))
        axs[ax].set_xticklabels(labels=np.linspace(0,sim.R,6, dtype=int), rotation=0)
        axs[ax].set_yticks(ticks=np.linspace(0,sim.R,6, dtype=int))
        axs[ax].set_yticklabels(labels=np.linspace(0,sim.R,6, dtype=int))
        axs[ax].set_aspect('equal', adjustable='box')
        axs[ax].set_title(r'bw=2')
        # axs[1].grid()
        # axs[1].minorticks_on()
        # axs[1].set_title("({})".format(titles[self.tit_cnt]), fontdict={'family':'times','weight': 'bold','size': 20},loc='right')
        # plt.title('Year {}'.format(sim.t), x=-0.5, y=1)
        # self.tit_cnt += 1
        plt.tight_layout()


        # self.fig, self.ax = plt.subplots(1)
        # self.x = self.X_c
        # self.y = self.Y_c
        # self.sc = visu.mscatter(s=self.s, c=self.c, edgecolors=self.edge, cmap=self.cmap)

        # self.annot = self.ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
        #                     bbox=dict(boxstyle="round", fc="w"),
        #                     arrowprops=dict(arrowstyle="->"))
        # self.annot.set_visible(False)

        # self.fig.canvas.mpl_connect("motion_notify_event", visu.hover)
        # plt.xlabel('X coord')
        # plt.ylabel('Y coord')
        # plt.legend(handles=self.sc.legend_elements()[0], labels=leg, loc='upper left', bbox_to_anchor=(1.05, 1))
        plt.title('Model @ year {}'.format(sim.t))
        # plt.savefig(sim.save_loc + '/spatial_dist-'+ sim.sim_id +'.png', dpi=400)

    def hover(self, event):
        # Ammended code from below:
        # https://stackoverflow.com/questions/7908636/possible-to-make-labels-appear-when-hovering-over-a-point-in-matplotlib 

        vis = self.annot.get_visible() 
        if event.inaxes == self.ax:
            cont, self.ind = self.sc.contains(event)
            if cont:
                visu.update_annot()
                self.annot.set_visible(True)
                self.fig.canvas.draw_idle()
            else:
                if vis:
                    self.annot.set_visible(False)
                    self.fig.canvas.draw_idle()
    
    def mscatter(self, **kw):
        ax = self.ax
        m = self.mark
        # Ammended code from below:
        # https://github.com/matplotlib/matplotlib/issues/11155
        if not ax: ax=plt.gca()
        sc = ax.scatter(self.x, self.y, **kw)
        if (m is not None) and (len(m)==len(self.x)):
            paths = []
            for marker in m:
                if isinstance(marker, mmarkers.MarkerStyle):
                    marker_obj = marker
                else:
                    marker_obj = mmarkers.MarkerStyle(marker)
                path = marker_obj.get_path().transformed(
                            marker_obj.get_transform())
                paths.append(path)
            sc.set_paths(paths)
        return sc
    
    def update_annot(self):
        # Ammended code from below:
        # https://stackoverflow.com/questions/7908636/possible-to-make-labels-appear-when-hovering-over-a-point-in-matplotlib 

        pos = self.sc.get_offsets()[self.ind["ind"][0]]
        self.annot.xy = pos
        text = "{}".format(" ".join([self.names[n] for n in self.ind["ind"]]))
        self.annot.set_text(text)

    def kdemap(self):
        """
        !! EXTRA VISUALISATION FUNCTION !!
        !!! NOT USED IN PAPER !!!

        kdemap() creates KDE plot of all input coordinates. 
        Can input by characteristic type, preference, utility thresholds.
        """

        cmaps = ['Blues', 'Reds', 'Greens']
        fig, axs = plt.subplots(len(self.types),1)

        for x in range(len(self.types)):
            loc = sim.r[:,1,0:2][np.array(sim.p[:,0,2][np.where(sim.p[:,1,2]== pop.q_bound['type I/II'][x])], dtype=int)]
            sns.kdeplot(loc, cmap=cmaps[x], shade=True, bw_adjust=.5, ax=axs[x])#, ax=axs[x+1,2])#, bins=[np.arange(0,500,10),np.arange(0,500,10)], cmap=plt.cm.Reds)

            axs[x].set_title('Type {} household'.format(pop.q_bound['type I/II'][x]))
            axs[x].set_xlabel('x (km)')
            axs[x].set_ylabel('y (km)')
            axs[x].set_xlim([0, dwell.R])
            axs[x].set_ylim([0, dwell.R])

        plt.tight_layout()
        # plt.savefig(sim.save_loc + '/kdemap-'+ sim.sim_id +'.png', dpi=400)

    def wealthmap(self):
        """
        !! EXTRA VISUALISATION FUNCTION !!
        !!! NOT USED IN PAPER !!!

        wealthmap() creates KDE plot of households filtered by their wealth. 
        Incrementing in 10% levels -> 10 plots.
        """

        cmaps = ['Blues']
        num_plots = 10
        fig, axs = plt.subplots(2,num_plots//2)    # this is just straight line, use 3x3 for collage
        axs = axs.flatten() # dont need this for flattened plot
        leg = ['Wealth ({}-{}%)'.format(int(100*(x)/num_plots), int(100*(x+1)/num_plots)) for x in range(num_plots)]
        self.mse_dists = []
        for x in range(num_plots):

            loc = sim.r[:,1,0:2][np.array(sim.p[:,0,2][int(x/num_plots*len(sim.p)):int(((x+1)/num_plots*len(sim.p))+1)], dtype=int)]
            sns.kdeplot(loc, cmap='Blues', shade=True, bw=2, ax=axs[x])#, ax=axs[x+1,2])#, bins=[np.arange(0,500,10),np.arange(0,500,10)], cmap=plt.cm.Reds)

            axs[x].set_title(leg[x])
            axs[x].set_xlim([0, dwell.R])
            axs[x].set_ylim([0, dwell.R])
            axs[x].set_aspect('equal', adjustable='box')

        # plt.savefig(sim.save_loc + '/wealthmap-'+ sim.sim_id +'.png', dpi=400)

    def U_all(self):
        """
        !! EXTRA VISUALISATION FUNCTION !!
        !!! NOT USED IN PAPER !!!

        U_all() scatters household utility as blue dot, plots smoothened utility as red.

        """

        plt.figure()
        U = np.array(sim.p[:,0,-1], dtype=float)
        U = U[U>-10]

        plt.scatter(np.linspace(1,len(U),len(U)), U, s=10, marker='.')
        U_smooth = gaussian_filter(U, sigma=3)

        plt.plot(U_smooth, color='crimson')
        plt.xlabel(r'$s_0$')
        plt.ylabel(r'$U$')
        plt.xticks(ticks=np.linspace(1,len(U_smooth),6), labels=np.round(np.linspace(-0.999,-0.001,6),2))
        # plt.savefig(sim.save_loc + '/U_all-'+ sim.sim_id +'.png', dpi=400)

    def plot_U_wealthy(self):
        """
        !! EXTRA VISUALISATION FUNCTION !!
        !!! NOT USED IN PAPER !!!
        """

        plt.figure()

        U_rich = []
        # poor = np.array([sim.p[:100,0,0]],dtype=int).T
        # poor = sim.p[:10,0,0]
        rich = np.array([sim.p[-20:,0,0]],dtype=int).T
        for x in sim.all_store_pop[4:]:
            U_rich.append(x[rich,0,-1])

            # U_poor.append(np.mean(x[poor,0,-1][0]))
            # U.append(x[np.where(sim.p[:,0,0]==6),0,-1][0][0][0])
        # U = sim.all_store_pop[0]#[:, int(np.where(sim.p[:,0,0]==576)[0])]

        # U = sim.all_store_pop[:, sim.p[np.where(sim.p[:,0,0]==576),0,-1]]

        U_rich = np.array(U_rich, dtype=float).T[0]
        # U_rich = U_rich[U_rich>0]

        # plt.plot(U_poor)
        for x in U_rich:
            plt.plot(x, alpha=0.3)
        # plt.legend(['poor','rich'])
        # plt.xticks(ticks=np.linspace(0,2,6),labels=np.linspace(0,25,6))
        plt.xlabel('t (years)')
        plt.ylabel('Utility')

        # plt.savefig(sim.save_loc + '/U_t_wealthiest'+ sim.sim_id +'.png', dpi=400)

    def U_specific_household(self):
        """
        !! EXTRA VISUALISATION FUNCTION !!
        !!! NOT USED IN PAPER !!!

        U_specific_household() plots the U of a specific household throughout time in the simulation.
        """

        plt.figure()
        U = []
        for x in sim.all_store_pop:
            U.append(x[np.where(sim.p[:,0,0]==576),0,-1][0][0][0])
        # U = sim.all_store_pop[0]#[:, int(np.where(sim.p[:,0,0]==576)[0])]
        # U = sim.all_store_pop[:, sim.p[np.where(sim.p[:,0,0]==576),0,-1]]
        U = np.array(U, dtype=float)
        plt.plot(U)
        # plt.xticks(ticks=np.linspace(0,2,6),labels=np.linspace(0,25,6))
        plt.xlabel('Timesteps')
        plt.ylabel('Utility')
        plt.show()

    def U_by_class(self):
        """
        !! EXTRA VISUALISATION FUNCTION !!
        !!! NOT USED IN PAPER !!!

        U_by_class() plots the average U by wealth class of households;
        in 20% groups. 
        """

        plt.figure()
        U_poor = []
        U_low_middle = []
        U_mid_middle = []
        U_up_middle = []
        U_rich = []

        poor = np.array([sim.p[:200,0,0]],dtype=int).T
        low_middle = np.array([sim.p[200:400,0,0]],dtype=int).T
        mid_middle = np.array([sim.p[400:600,0,0]],dtype=int).T
        up_middle = np.array([sim.p[-400:-200,0,0]],dtype=int).T
        rich = np.array([sim.p[-200:,0,0]],dtype=int).T

        for x in sim.all_store_pop:
            U_rich.append(np.mean(x[rich,0,-1][0]))
            U_low_middle.append(np.mean(x[low_middle,0,-1][0]))
            U_mid_middle.append(np.mean(x[mid_middle,0,-1][0]))
            U_up_middle.append(np.mean(x[up_middle,0,-1][0]))
            U_poor.append(np.mean(x[poor,0,-1][0]))

        U_poor = np.array(U_poor, dtype=float)
        U_poor = U_poor[abs(U_poor - np.mean(U_poor)) < 4 * np.std(U_poor)]
        U_low_middle = np.array(U_low_middle, dtype=float)
        U_low_middle = U_low_middle[abs(U_low_middle - np.mean(U_low_middle)) < 4 * np.std(U_low_middle)]
        U_mid_middle = np.array(U_mid_middle, dtype=float)
        U_mid_middle = U_mid_middle[abs(U_mid_middle - np.mean(U_mid_middle)) < 4 * np.std(U_mid_middle)]
        U_up_middle = np.array(U_up_middle, dtype=float)
        U_up_middle = U_up_middle[abs(U_up_middle - np.mean(U_up_middle)) < 4 * np.std(U_up_middle)]
        U_rich = np.array(U_rich, dtype=float)
        U_rich = U_rich[abs(U_rich - np.mean(U_rich)) < 4 * np.std(U_rich)]

        # USE NP.WHERE FOR THE ABOVE!!!!!

        plt.plot(U_poor)
        plt.plot(U_low_middle)
        plt.plot(U_mid_middle)
        plt.plot(U_up_middle)
        plt.plot(U_rich)
        plt.legend(['0-20%','20-40%','40-60%','60-80%','80-100%'])
        plt.xticks(ticks=np.linspace(0,sim.t*sim.t_inc,6), labels=np.linspace(0,sim.t,6, dtype=int))

        plt.xlabel('Years')
        plt.ylabel('Utility')   
        # plt.savefig(sim.save_loc + '/U_by_class-'+ sim.sim_id +'.png', dpi=400)

    
    def B0_dwell(self):
        """
        !! EXTRA VISUALISATION FUNCTION !!
        !!! NOT USED IN PAPER !!!

        B0_dwell() plots the B0 of a specific dwelling throughout time in the simulation.
        """

        d = 0

        plt.figure()
        B0 = []
        for x in sim.all_store_dwell:
            B0.append(x[d,2,0])

        occ_B0 = []
        tmp_B0 = []
        occ_t = []
        tmp_t = []
        on = 0
        for y, x in enumerate(sim.all_store_dwell):
            if x[d,0,1] != 0:
                tmp_B0.append(x[d,2,0])
                tmp_t.append(y)
                on = 1
            elif x[d,0,1] == 0 and on==1:
                occ_B0.append(tmp_B0)
                occ_t.append(tmp_t)
                tmp_B0 =[]
                tmp_t =[]
                on = 0

        plt.plot(B0)

        for x, y in zip(occ_t, occ_B0):
            plt.plot(x, y, color='crimson')

        plt.xlabel('t (years)')
        plt.ylabel('B0 (specific dwelling)')
        # plt.savefig(sim.save_loc + '/B0_dwelling={}-'.format(d)+ sim.sim_id +'.png', dpi=400)

    def B0_household(self):
        """
        !! EXTRA VISUALISATION FUNCTION !!
        !!! NOT USED IN PAPER !!!

        B0_household() plots the B0 of dwellings occupied by specific households through the time in simulation.
        """

        plt.figure()

        pp = [10,20,30,40,50]
        for p in pp:
            B0 = []
            for y,x in enumerate(sim.all_store_pop):
                d = int(x[p,0,2])
                r = sim.all_store_dwell[y]
                
                B0.append(r[d,2,0])

            plt.plot(np.linspace(1,sim.t,len(B0)), B0)
        plt.xlabel('t (years)')
        plt.ylabel('B0 (specific household)')
        # plt.savefig(sim.save_loc + '/household={}-'.format(pp)+ sim.sim_id +'.png', dpi=400)

    def scatter_unocc(self):
        """
        !! EXTRA VISUALISATION FUNCTION !!
        !!! NOT USED IN PAPER !!!

        scatter_unocc() creates a scatter plot of the price B_0 of occupied and unoccupied dwellings through time.
        """

        plt.figure()
        B0 = []
        for x in sim.all_store_dwell:
            occ = np.where(x[:,0,1]==0)
            B0.append(x[occ,2,0][0])

        for y,x in enumerate(B0):
            plt.scatter([y+1]*len(x), x, s=10, marker='.',c='blue')
        plt.xlabel('t (years)')
        plt.ylabel('B0 (occupied dwellings)')
        # plt.savefig(sim.save_loc + '/B0_t_occupied-' + sim.sim_id +'.png', dpi=400)

        plt.figure()
        B0 = []
        for x in sim.all_store_dwell:
            un_occ = np.where(x[:,0,1]!=0)
            B0.append(x[un_occ,2,0][0])

        for y,x in enumerate(B0):
            plt.scatter([y+1]*len(x), x, s=10, marker='.',c='red')
        plt.xlabel('t (years)')
        plt.ylabel('B0 (unoccupied dwellings)')
        # plt.savefig(sim.save_loc + '/B0_t_unoccupied-' + sim.sim_id +'.png', dpi=400)

    
    def mean_B0_U(self):
        """
        !! EXTRA VISUALISATION FUNCTION !!
        !!! NOT USED IN PAPER !!!

        mean_B0_u() plots the mean dwelling B0 (occupied within max past 4 periods) vs time t,
        & plots the mean household U (all households) vs time t.
        """

        plt.figure()
        B0 = [0]
        for x in sim.all_store_dwell:
            occ = np.where(x[:,0,1]<=4)
            B0.append(np.mean(x[occ,2,0][0]))
        plt.plot(B0)
        plt.ylabel('Mean B0')
        plt.xlabel('Years')
        plt.xticks(ticks=np.linspace(0,sim.t*sim.t_inc,10, dtype=int), labels=np.linspace(0,sim.t,10, dtype=int))
        plt.savefig(sim.save_loc + '/B0_t-' + sim.sim_id +'.png', dpi=400)

        plt.figure()
        U = [0]
        for x in sim.all_store_pop:
            U.append(np.mean(x[:,0,-1]))
        plt.plot(U)
        plt.ylabel('Mean U')
        plt.xlabel('Years')
        plt.xticks(ticks=np.linspace(0,sim.t*sim.t_inc,10, dtype=int), labels=np.linspace(0,sim.t,10, dtype=int))
        # plt.savefig(sim.save_loc + '/U_t-'+ sim.sim_id +'.png', dpi=400)

    def B0_M_animate(self):
        """
        !! EXTRA VISUALISATION FUNCTION !!
        !!! NOT USED IN PAPER !!!

        B0_M_animate() plots the distribution of households M by their B0 (left) and their s0 (right) through time; animated plot.
        """
        
        while True:
            for x in range(len(sim.all_store_pop)):
                fig = plt.figure()
                ax = fig.add_subplot(111)

                simp = sim.all_store_pop[x]
                simr = sim.all_store_dwell[x]

                where = np.array(simp[:,0,2],dtype=int)
                occ = np.array(simp[:,0,0], dtype=int)
                B0 = np.array(simr[where,2,0])
                B0_occ = np.array(sorted(zip(B0,occ), key=lambda x:x[0]))
                B0 = B0_occ[:,0]
                occ = np.array([B0_occ[:,1]], dtype=int).T

                ax.scatter(np.linspace(1,len(B0),len(B0)), B0, s=10, marker='.', c='blue')

                ax2 = ax.twinx()
                s0 = np.array(simp[occ,2,0])
                ax2.scatter(np.linspace(1,len(s0),len(s0)), s0, s=10, marker='.', c='crimson')

                ax.set_xlabel(r'$M$')
                ax.set_ylabel(r'$B_0$')
                ax2.set_ylabel(r'$s_0$')
                fig.legend(['B0','s0'], loc='left')
                plt.title('Year {}'.format(x))
                plt.pause(0.1)
                plt.close()
    
    def liveplot(self):
        """
        !! EXTRA VISUALISATION FUNCTION !!
        !!! NOT USED IN PAPER !!!
        
        this will load up all saved data from the respective folder then iterate through as a 'live plot' 
        """

        while True:
            for x in range(len(sim.all_store_pop)):
                sim.step = (x+1)*sim.t_inc
                sim.p = sim.all_store_pop[x]
                sim.r = sim.all_store_dwell[x]

                visu.spatial_dist()
                plt.pause(0.02)
                plt.close()
            plt.show()

    def interactive_spatial(self):
        """
        !! EXTRA VISUALISATION FUNCTION !!
        !!! NOT USED IN PAPER !!!
        
        interactive_spatial() creates a scatter plot of dwellings and households, and displays info of characteristics, preferences, and utilities of the 
        corresponding household/dwelling upon mouse hovering over the point.
        """
        sim.step = 'END'
        # Ammended code from below:
        # https://stackoverflow.com/questions/7908636/possible-to-make-labels-appear-when-hovering-over-a-point-in-matplotlib 

        X_r, Y_r = sim.r[:,1][:,0:2].T      # coordinates of dwelling

        sim.p = np.array(sorted(sim.p, key=lambda x:x[2][0]))   # sort by wealth
        # sim.p = np.array(sorted(sim.p, key=lambda x:x[1][2]))   # sort by household type
        types = [sum([1 for x in sim.p[:,1,2] if x == pop.q_bound['type I/II'][y]]) for y in range(len(pop.q_bound['type I/II'])) ]
        types = [len(sim.p)]
        self.types = types

        # colours = ['midnightblue','crimson','forestgreen']
        colours = ['Black','crimson','forestgreen']

        self.edge = ['lightgrey']*len(X_r)
        self.c = [0]*len(X_r)

        for y, x in enumerate(types):
            self.edge.extend([colours[y]]*x)
            self.c.extend([y+1]*x)

        cmap = ['lightgrey'] + colours[:len(types)]
        leg = ['Dwellings'] + ['Type {}'.format(x+1) for x in range(len(types))]
        self.cmap = ListedColormap(cmap)

        X_p, Y_p = sim.r[:,1,0:2][np.array(sim.p[:,0,2], dtype=int)].T   # coordinates of each household

        self.r_names = ["U0: {} \n r: ({}, {}) \n B: {} \n -------- \n".format(round(x[0][2],1), x[1][0],x[1][1], [round(y,1) for y in x[2][:3]]) for x in sim.r]
        self.p_names = ["p: {}, U: {}, n_T: {}, r_ID: {} \n q: {} \n s: {}\n G: {}".format(x[0][0], round(x[0][4][0],1), x[0][3], x[0][2], [round(y,3) for y in x[1]], \
                        [round(y,3) for y in x[2][:3]], [round(y,3) for y in x[3][:3]]) for x in sim.p]

        self.names = np.concatenate((self.r_names, self.p_names),axis=0)
        self.X_c = np.concatenate((X_r, X_p), axis=0)
        self.Y_c = np.concatenate((Y_r, Y_p), axis=0)
        self.s = [10]*len(X_r) + [10]*len(X_p)
        self.mark = [str(".")]*len(X_r) + [str(".")]*len(X_p)
        # self.edge = ['C7']*len(X_r) + ['C3']*len(X_p)
        # self.c = [0]*len(X_r) + [1]*len(X_p)
        # self.cmap = ListedColormap(['lightgrey', 'C3'])
        

        self.fig, self.ax = plt.subplots(1)
        self.x = self.X_c
        self.y = self.Y_c
        self.sc = visu.mscatter(s=self.s, c=self.c, edgecolors=self.edge, cmap=self.cmap)

        self.annot = self.ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="w"),
                            arrowprops=dict(arrowstyle="->"))
        self.annot.set_visible(False)
        self.fig.canvas.mpl_connect("motion_notify_event", visu.hover)


if __name__ == '__main__':

    start = time.time()
    global sim
    sim = simulate()    # initialise the model
    sim.create()        # create households and dwellings
    sim.visualise_results()    # load from save and carry out visualisation