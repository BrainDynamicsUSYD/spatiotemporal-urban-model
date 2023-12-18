import numpy as np
np.random.seed(69)


import pytz
import time
from datetime import datetime

from matplotlib import cm, pyplot as plt
import matplotlib.markers as mmarkers
from matplotlib.colors import ListedColormap
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.ndimage.filters import gaussian_filter

from os.path import dirname, abspath
import os
import sys
import shutil
global dirmain, diroutput     # relative directories
dirmain = dirname(abspath(__file__))
diroutput = dirmain + '/data/output/'
from market_dynamics_visualisations import visualisation 

"""
Alexander McInnes, Jun 2021
alexmcinnes97@gmail.com
alexander.mcinnes@sydney.edu.au
This project is licensed under the GNU General Public License Version 3 - see the LICENSE.txt file for details.

*** NOT TO BE SHARED UNTIL PUBLICATION OF PAPER ***

directory setup:
model > market_dynamics.py (main)
      > md_visualisations.py (secondary to plot all results)
      > data  > output (outputted results of simulation)  > sim-ID > households_sim-ID_timestep.txt
                                                                   > dwellings_sim-ID_timestep.txt            
              > input (used for visualisations after from output results after simulation completed)

Typical workflow:
1. specify case to run in __name__
2. specify params of case/model in simulate.model_parameters()
3. results output, results visualised/plotted
optional 4. in md_visualisation.py, comment out plots to create, run model
"""

class simtime:
    def current():
        """
        current() converts current time to format of yearmonthdayhourminutesecond for the ID of simulation.

        return: simulation ID final
        """

        dtT = datetime.now(pytz.timezone('Australia/Sydney'))
        final = [dtT.strftime("%"+str(x)) for x in 'ymdHMS']
        final=int(''.join(map(str, final)))

        return final

class household:
    def __init__(self, N, N_min):
        """ 
        __init__ initialises parameters and dictionaries for which characteristics and preferences are sampled from later.
        
        param N: input number of households 
        param N_min: absolute minimum number of households 

        """

        self.N = int(N)  
        self.N_min = int(N_min)       
        self.q_bound = {'n_people': np.linspace(1,4,4), 'type I/II': [-1,1]}   # bounds for characteristics, 
                                            # type I/II is for testing different characteristics of households
        self.s_bound = {'s_0': [-1,0], 'classical:modern': [-1, 1], 'large house': [0,1],'s1 preference':[-1,1]}   # bounds for preferences

    def populate(self):
        """ 
        populate() generates households, their characteristics and their preference strengths for dwelling characteristics.
            Array data can be either generated or loaded from previous run.

        field self.s: preference vector of households [s_0, ...]
        field self.q: characteristic vector of households [q_0, ...]
        field self.U: utility of households
        field self.p: general household info [household ID, on/off-market flag, occupied dwelling ID, n times transacted]
                in saving household array, self.p will contain utility values at end.

        """

        q = pop.characteristics(self.N)     # household characteristics q 
        
        self.s_0 = calcs.price_sens_income_dist()   # price sensitivity s_0
        self.s_ = pop.preferences(self.N)       # preferences s_ is the s' vector in Paper 1. in this code it also contains s_0 for ease
        self.q = q                              # group all the household characteristics

        self.s = pop.normalise_prefs(s_0=self.s_0, s_=self.s_)      # normalise preference vector s by s_0
        if sim.case != 'C':
            self.s = self.s_0                   # restrict to s0 only

        self.U = pop.utilities(self.N)      # household utilities U
        self.p = pop.general(self.N)        # general household info p

    def characteristics(self, num):
        """
        characteristics(num) generates an N*num size characteristic vector for num households and m+1 characteristics;
        characteristics are sampled from initialized dictionary within __init__.

        param num: length of characteristic vector to generate, i.e. no. households to generate for
        return: concatenated characteristic vector returned, size (m+1)*num
        """

        q_0 = np.ones(shape=(num, 1))       # characteristic for clustering
        q_1 = np.sort(np.random.choice(self.q_bound['n_people'], size=(num, 1)), axis=0)/4   # number of occupants in each household
        q_2 = np.random.choice(self.q_bound['type I/II'], size=(num, 1))    # this is a sample test case characteristic
        if sim.case == 'D':
            q_3 = np.random.choice(self.q_bound['type I/II'], size=(num, 1))    # this is a sample test case characteristic
            q = np.concatenate((q_0, q_3), axis=1)  # group all the household characteristics
        else:
            q = q_0

        return np.round(np.array(q, dtype=float), 3)

    def preferences(self, num):
        """
        preferences(num) generates an N*num size preference vector for num households and m+1 preferences;
        preferences are sampled from initialized dictionary within __init__.

        param num: length of preference vector to generate, i.e. no. households to generate for
        return: concatenated preference vector returned, size (n+1)*num

        ** Later can instantiate random preferences by different dynamics/distributions. 
        """

        s_1 = 1 - 2*np.random.random_sample(size=(num, 1))   # random preferences for classical (-1)/modern (+1) dwelling 
        s_2 = np.sort(np.random.random_sample(size=(num, 1)), axis=0)    # random preferences for large house
        s_3 = 1 - 2*np.random.random_sample(size=(num, 1))     # random preferences to live near factories vs. sea

        s_ = np.concatenate((s_1, s_2), axis=1)     # group all the preference strengths 
        
        if sim.case == 'C':
            s_ = s_3    # preference for dwelling characteristic B1 
            # s_ = np.concatenate((s_1, s_3), axis=1)     # group all the preference strengths 

        return np.array(s_, dtype=float)

    def normalise_prefs(self, **kwargs):
        """ 
        normalise_prefs(s_0, s_) normalises the input strength of preferences s_ as according to |s_0|^2 + s_^2 = 1.

        param s_0: price sensitivity of households on range (-1, 0), bounding parameter in normalisation
        param s_: vector s'(p) containing relative strengths of preferences for various dwelling features.        
        return: normalised preference vector s containing [s_0, s_1,... , s_n]
        """

        if len(kwargs.items())==2:
            for key,val in kwargs.items():
                exec('self.'+key + '=val')

        elif len(kwargs.items())==1:
            for key,val in kwargs.items():
                exec('self.'+key + '=val')
            self.s_0 = np.array([self.s_n[:,0]]).T
            self.s_ = np.array(self.s_n[:,1:])


        s_norm = [1-abs(x[0])**2 for x in self.s_0]     # normalise the strength of preferences as according to |s_0|^2 + s'^2 = 1 
        s_sum = [sum(x**2) for x in self.s_]     # s_sum (total preferences strength vector) must equal s_norm (for normalisation)

        s_new = []
        for x in range(len(s_norm)):
            s_new.append(self.s_[x]*np.sqrt(s_norm[x]/s_sum[x]))

        s = np.concatenate((np.array(self.s_0), s_new), axis=1)      # join s_1,2,3.. to s_0 array
        s = np.nan_to_num(s, nan=0)

        return np.round(np.array(s, dtype=float), 3)

    def utilities(self, num):
        """ 
        utilities(num) generates an empty num*1 size array to store utilities of households.

        param num: length of utility vector to generate, i.e. no. households to generate for
        return: empty utility vector U size num*1
        """

        U = np.zeros(shape=(num, 1), dtype=object)

        return U

    def general(self, num):
        """ 
        general(num) generates an empty array for storing general information of households. 

        param num: length of general info vector to generate, i.e. no. households to generate for    
        return: empty array p size num*4, [household ID, on/off-market flag, occuping dwelling ID, no. transactions]
        """

        p_0 = np.linspace(0, num-1, num, dtype=int)   # household ID
        p_1 = np.ones(shape=(num, 1), dtype=int)    # household on-market flag
        p_2 = np.zeros(shape=(num, 1), dtype=int)   # dwelling occupied by household
        p_3 = np.zeros(shape=(num, 1), dtype=int)   # no. transactions by household

        p = np.concatenate((np.array([p_0]).T, p_1, p_2, p_3), axis=1)   # compile all general household information

        return p

    def assign_dwellings(self):
        """ 
        assign_dwellings() assigns dwellings to initialised population. 

        Current method: random assignment

        Alternatively, could assign by order of increasing wealth of households to dwellings with positive, ascending U0 values.
            i.e., poorest household will be assigned a dwelling with positive U0 closest to 0,
                2nd poorest to a dwelling with positive U0 & 2nd closest to 0 etc.

        """

        """ ASSIGN VIA ORDER OF U0 & s0 -- not used         """
        # s_mark = self.s[self.p[:,1]>0]
        # on_mkt = len(s_mark)

        # s_sort = np.array(sorted(enumerate(s_mark[:,0]), key=lambda x:x[1]), dtype=int)[:,0]
        # B_sort = np.array(sorted(dwell.r[dwell.r[:,2]>0], key=lambda x:x[2]), dtype=int)[:len(s_sort)]
        # B_sort = np.array(dwell.r[dwell.r[:,2]>0], dtype=int)[:len(s_sort)]

        # if len(B_sort) != len(s_sort):
        #     B_sort = np.array(sorted(dwell.r, key=lambda x:x[2]), dtype=int)[:len(s_sort)]

        # U = np.array([dwell.r[B_sort[:,0]][:,2] + np.sum(self.s[s_sort] *dwell.B[B_sort[:,0]], axis=1)]).T
        # self.U[-on_mkt:] += U #- abs(U)*calcs.fees     # assign utility for random dwelling
        # self.p[-on_mkt:,1] = 0     # set off-market
        # self.p[-on_mkt:,2] = B_sort[:,0]   # assign dwelling 
        # dwell.r[:,1][B_sort[:,0]] = 0   # set dwelling off-market
        """#################################################"""

        """ ASSIGN VIA RANDOM """
        on_mkt = len(self.s[self.p[:,1]>0])
        r_mark = np.array(dwell.r[np.where(dwell.r[:,1]!=0)][:,0], dtype=int)
        dwell_id = np.random.choice(r_mark, on_mkt, replace=False)

        self.p[-on_mkt:,2] = dwell_id           # assign dwelling 
        self.p[-on_mkt:,1] = 0                  # set household off-market
        dwell.r[dwell_id,1] = 0                 # set dwelling off-market
        """####################################################"""

        """ update U & B0 for all households following assignment """
        # household data
        p = self.p
        U = self.U
        q = self.q
        s = self.s

        # corresponding dwelling data for each household occupying it
        r = dwell.r[p[:,2]]
        U0 = r[:,2]
        r_coord = dwell.r_coord[p[:,2]]
        B = dwell.B[p[:,2]]

        U_new = U0 + np.sum(s*B, axis=1)    # updated utility for their current dwelling

        # calculate euclidean distance between all households
        loc = dwell.r_coord[p[:,2], 0:2]
        dwell_1 = np.reshape(np.repeat(loc, len(loc), axis=1).T, (2,-1)).T
        dwell_2 = np.reshape(np.repeat(loc, len(loc), axis=0).T, (2,-1)).T
        euclids = np.reshape(np.linalg.norm(dwell_1.T - dwell_2.T, axis=0), (len(loc), -1))

        # calculate final term of the utility equation for different types of households.
        q = np.array(self.q)
        G = sim.G_0*np.exp(-euclids**2/sim.h**2)
        U_G = np.sum(np.dot(q, q.T)* G, axis=1) * dwell.price_scale

        # temp store utility to assign price B0 to dwelling
        tmp_U = np.round(np.array([U_new + U_G], dtype=float).T, 3)

        # recalculate B0 from U' = U - s0*B0
        if np.shape(dwell.B[p[:,2]])[1] == 1:
            dB = tmp_U - self.s*dwell.B[p[:,2]]
        else:
            dB = tmp_U - np.array([self.s[:,0]*dwell.B[p[:,2], 0]]).T

        # assign price B0 to occupied dwellings
        dwell.B[p[:,2],0] = np.round(np.array(dB.T, dtype=float), 3)

        # recalculate household utility for assigned price
        U_new = U0 + np.sum(s*dwell.B[p[:,2]], axis=1)
        self.U = np.round(np.array([U_new + U_G], dtype=float).T, 3)

    def perturb(self):
        """
        perturb() will alter preferences and/or characteristics of households. 
        Can include age of household/shift preferences with age.

        alter household preferences by eq. 17: rotate s' vector in n-dimensional subspace by taking mean of preferences of
        households within local neighbourhood.

        s = -alpha * [s(p,t) - <s(p',t)>] * ∆t

        """

        s = pop.s
        if np.shape(s)[1] >1:
            p = pop.p

            # calculate euclidean distance between all households
            loc = dwell.r_coord[p[:,2], 0:2]
            dwell_1 = np.reshape(np.repeat(loc, len(loc), axis=1).T, (2,-1)).T
            dwell_2 = np.reshape(np.repeat(loc, len(loc), axis=0).T, (2,-1)).T
            euclids = np.reshape(np.linalg.norm(dwell_1.T - dwell_2.T, axis=0), (len(loc), -1))

            # carry out eq. 17, iterate through households
            s_n = []
            for h in range(len(p)):
                d = euclids[h]          # distance of other households from household h
                m = np.where(d<=sim.h)[0]           # households within local neighbourhood of h
                s_m = np.mean(s[m],axis=0)          # take mean of local preferences

                s_h = -sim.alpha*(s[h] - s_m)       # calculate incremental change to preferences via eq. 17
                s_tmp = s[h] + s_h      # add incremental change to their preferences
                s_tmp[0] = s[h,0]       # reset first element to their static income sensitivity

                s_n.append(s_tmp)

            pop.s = pop.normalise_prefs(s_n=np.array(s_n))    # normalise and reassign perturbed preferences

        # re-calculate income distribution and wealth
        # household.mng_wealth(self)

    def mng_wealth(self):
        """ 
        mng_wealth() will shift income, emulate wage growth, recalculate price sensitivity s_0
        
        """
        # in early versions maybe just upshift income with age (generally) and lower price sensitivity, 
        # but negatively perturb the price sensitivity of random households so that the income distribution remains relatively static. 
        # i.e. can fluctuate slightly as noise but not correlated with time.
        # as people are removed, new households with same price sensitivity or higher price sensitivity such that static distribution is maintained

        #this will also include wage/wealth growth etc. Calculate current wealth from a pre-defined networth as a function of age
        pass

    def evolve(self):
        """ 
        evolve() contains methods relating to manipulation of the total number of households 
            i.e. altering demand via increasing/decreasing number of households.
        Emulates population growth/decline (birth, death, marriage, immigration, emmigrate, moving out etc.)
        
        Method for increasing calls functions used in the populate() function during initialisation of the households class (pop), 
            creates 'new' number of households.

        Method for decreasing deletes 'demo' number of random households, setting their occupied dwellings on-market.

        Both methods re-adjust the household IDs to correspond with the total number of households in the model.

        """

        """ HOUSEHOLD CREATION """
        new = int(sim.eps*self.N)   # no. households to create
        # check for upper bound on creation
        if self.N + new <= dwell.M:
            pass
        else:
            new = dwell.M - self.N

        if new > 0 and sim.step/4>40:       # threshold to create households, if any & after time period t
            if hasattr(sim, 'new_min') == False:
                sim.new_min = sim.step  # timestep at which household creation begins

            if self.N + new <= dwell.M:
                pop.growth(new)

        elif sim.eps!=0 and new == 0 and sim.step/4>40 and hasattr(sim, 'new_max') == False:
            sim.new_max = sim.step  # timestep at which household creation ceases

        """ HOUSEHOLD DESTRUCTION """
        demo = int(sim.rho*self.N)   # no. households to destroy

        # check for lower bound on destruction
        if self.N - demo > self.N_min: 
            pass
        else:
            demo = self.N - self.N_min

        if demo > 0 and sim.step/4>40:
            if hasattr(sim, 'new_min') == False:
                sim.new_min = sim.step  # timestep at which household creation begins

            if self.N -demo >= self.N_min:
                pop.destroy(demo)
        elif sim.rho!=0 and new == 0 and sim.step/4>40 and hasattr(sim, 'new_max') == False:
            sim.new_max = sim.step  # timestep at which household creation ceases

    def growth(self, new):
        """
        growth() generates new housholds to add to existing households. Characteristics, preferences, and all other info
        is generated as per the populate() function.

        param new: input number of households to create

        """

        dq = pop.characteristics(new)
        ds_0 = calcs.sample_s0_income_dist(new)
        ds_ = pop.preferences(new)

        ds = pop.normalise_prefs(s_0=ds_0, s_=ds_)
        dU = pop.utilities(new)
        dp = pop.general(new)
        dp[:,0] += np.shape(self.p)[0]

        # check shape of s vector and q vector
        if np.shape(self.s)[1] == 1:
            ds = np.array([ds[:,0]], dtype=float).T
        if np.shape(self.q)[1] == 1:
            dq = np.array([dq[:,0]], dtype=float).T

        # concatenate all new households with existing
        self.s = np.concatenate((self.s, ds), axis=0)
        self.U = np.concatenate((self.U, dU), axis=0)
        self.p = np.concatenate((self.p, dp), axis=0)
        self.q = np.concatenate((self.q, dq), axis=0)

        # un_occ_dwell = dwell.r[dwell.r[:,1]!=0]
        # if np.size(un_occ_dwell) >0:
        pop.assign_dwellings()      # assign new households to dwellings
        # else:           # this else statement is for if there are no available dwellings for households
        """ ADJUST THIS IF REQUIRED - capacity for households to remain 'dwelling-less' i.e. homeless""" 
            # self.p[-new:,1] = 1     # set leave household on-market
            # self.p[-new:,2] = -1    # set household occupied dwelling to -1
        self.N = len(self.U)    # re-assign total number of households to pop.N 

    def destroy(self, demo):
        """
        destroy() removes a number of households from the model, setting previously occupied dwellings on-market.

        param demo: input number of households to remove

        """

        del_ind = np.random.choice(range(self.N), size=(demo,1), replace=False)    # index of dwellings to delete

        dwells = self.p[del_ind,2]  # dwellings occupied by households to delete
        dwell.r[dwells,1] = 1       # set occupied dwellings to on-market

        # delete all indices of households to be destroyed
        self.s = np.delete(self.s, del_ind, axis=0)
        self.U = np.delete(self.U, del_ind, axis=0)
        self.p = np.delete(self.p, del_ind, axis=0)
        self.q = np.delete(self.q, del_ind, axis=0)

        self.p[:,0] = np.linspace(0,len(self.s)-1, len(self.s))     # re-number households after deletion
        self.N = len(self.U)        # re-assign total number of households to pop.N 

class dwelling:
    def __init__(self, M, R, u0_max, u0_wid, price_scale, M_min, M_max, cent):
        """ 
        __init__ initialises parameters and dictionaries for which characteristics are sampled from later.
        
        param M: input number of dwellings 
        param R: linear size of model
        param u0_max: maximum U0 value 
        param u0_wid: window size for U0 distribution
        param price_scale: self explanatory by name, value to scale model values to; default unity scale
        param M_min: min number of dwellings permitted

        """

        self.M = int(M)    
        self.R = R    
        self.r_coord = np.zeros((1, 2))     # blank coordinate array to be filled during initialisation
        self.u0_max = float(u0_max)
        self.u0_wid = u0_wid
        self.price_scale = price_scale   
        self.M_min = M_min
        self.M_max = M_max
        self.cent = cent
        self.r_recs =np.array([])

        """ establish some dictionaries for later (will search by index of element to find same term to compare 
        utility of specific characteristics of dwelling) """

        # formate for dwelling characteristics, possible to extend
        self.style = {'classical:modern': [-1,1]}

    def construct(self):
        """ 
        construct() generates dwellings, their location, characteristics and other general information.
            Array data can be either generated or loaded from previous run.

        field self.B: characteristic vector of dwellings [B_0, ...]
        field self.r_coord: spatial location vector of dwellings [x, y, z, ...]
        field self.r: general dwelling info [dwelling ID, on/off-market flag, general utility U_0]

        """

        self.r_coord = dwell.coordinates(self.M)    # coordinates r of dwellings
        self.u_0 = dwell.utility_0(self.M)          # general utility U_0(r) of dwellings
        B = dwell.characteristics(self.M)           # characteristic vector B(r) of dwellings

        self.B = np.round(np.array(B*self.price_scale, dtype=float), 3)
        if sim.case != '999':
            self.B = np.array([self.B[:,0]]).T   # restrict to B0

        r = dwell.general(self.M)
        self.r = np.concatenate((r, self.u_0), axis=1)   # compile general information of dwelling [ID, on-market flag, U_0]

        if sim.case == 'C':
            dwell.B1_gen()

    def utility_0(self, num):
        """ 
        utility_0(num) generate general utility of dwellings, common to all households.

        This function also generates the distribution of the general utility U0 of dwellings.
            This method assigns either all null (ones, equivalently negligible) or via a distribution with n central peaks.
            Central peaks defined by decimal location of maximal size of model (0.5 is center).
            
        param num: length of general utility vector to generate, i.e. number of dwellings to generate U_0 for
        return: general utility vector returned, size 1*num.

        """

        centric = np.array(self.cent) * self.R  # U0 centricity location
        self.centric = centric

        if np.size(centric)>0:
            u_0 = np.ones(shape=(num,1), dtype=object) * self.price_scale

            # remove specified centricity coordinate if it is in coordinate list then move to start
            for x, cent in enumerate(centric):
                loc = np.where(np.all(self.r_coord==cent, axis=1))[0]
                if len(loc) > 0:
                    self.r_coord = np.delete(self.r_coord, loc, axis=0)
                    self.r_coord = np.insert(self.r_coord, x, cent, axis=0)
                else:
                    self.r_coord[x] = cent

            # check for duplicate coordinates, exit on error if duplicates
            uniq = np.unique(self.r_coord, axis=0)
            if len(uniq) != self.M:
                print("""ERR: len(uniq) != self.M
                        error in setting U0 centricity -- shutting down""")
                exit()

            """ down-scale all other utilities with distance from the central point(s) """

            # this generates a mapping of coordinates (1/4 integer approx.)
            self.incs = self.R*4 + 1
            dim = np.linspace(0, self.R, self.incs)
            x_map, y_map = np.meshgrid(dim, dim)
            u0_map = np.concatenate((np.array([x_map.flatten()]).T, np.array([y_map.flatten()]).T), axis=1)
            pos = np.empty(x_map.shape + (2,))
            pos[:, :, 0] = x_map
            pos[:, :, 1] = y_map

            # gaussian shape U0 distribution
            euclid_cents = []
            for x in range(len(centric)):
                euclid_cents.append(np.linalg.norm(u0_map.T - np.array([centric[x]]).T, axis=0))
            x = np.min(euclid_cents, axis=0)
            u0 = self.u0_max*np.exp(-abs(x**2)/(self.u0_wid**2))

            self.U0_store = np.reshape(u0, (self.incs,-1))
            r_coord_ref = np.array(np.round(self.r_coord * 4), dtype=int)   # cell reference for U0_map for coords
            for x in range(len(u_0)):
                u_0[x] = u_0[x]* self.U0_store[r_coord_ref[x,0]][r_coord_ref[x,1]]
            
        else:
            u_0 = np.zeros(shape=(num,1), dtype=object) 

        self.u_0 = u_0

        return np.array(u_0, dtype=object)

    def characteristics(self, num):
        """
        characteristics(num) generates an (n+1)*num size characteristic vector for num dwellings and n+1 characteristics;
        characteristics can be sampled from initialized dictionary within __init__.

        param num: length of characteristic vector to generate, i.e. no. dwellings to generate for
        return: concatenated characteristic vector returned, size (n+1)*num matching household preference vector size.
        """

        # B_0 = np.array(sorted(dwell.B0_sample(num)),dtype=float)
        B_0 = np.zeros(shape=(num,1))
        B_1 = 1 - 2*np.random.random_sample(size=(num, 1))  # style of building; classical (-1), modern (+1)
        B_2 = np.random.choice(pop.q_bound['n_people'], size=(num, 1))/4    # number of bedrooms in house (1-4 to match size of households) 

        B = np.concatenate((B_0, B_1, B_2), axis=1)  # compile all the dwelling characteristics 
        
        return np.round(np.array(B, dtype=float), 3)

    def B0_spatial_sample(self, num, dr_coord):
        """ 
        ***NOT USED***
        --- MAYBE USE THIS FUNCTION FOR SAMPLING CHARACTERISTICS IN DWELLING CREATION ?? 
        """
        B0 = np.zeros(shape=(num,1))

        age = 4
        inc = 1
        sz = self.R*inc + 1

        locs = np.zeros(shape=(sz,sz))
        w = np.zeros(shape=(sz,sz))

        for x in np.where(self.r[:,1]<=age)[0]:
            locs[int(np.round(self.r_coord[x,0])*inc)][int(np.round(self.r_coord[x,1])*inc)] += self.B[x,0]
            w[int(np.round(self.r_coord[x,0])*inc)][int(np.round(self.r_coord[x,1])*inc)] += 1

        locs /= w
        locs[np.isnan(locs)] = 0
        mean_grid = np.array(locs)

        sig = 1
        smooth_mean = gaussian_filter(mean_grid, sigma=sig)
        # smooth_mean /= np.max(smooth_mean)
        # smooth_mean *= np.max(self.B[np.where(self.r[:,1]<=age),0])

        # if sim.step%4==0:
        #     X,Y = np.meshgrid(np.arange(0,dwell.R*2+1,1),np.arange(0,dwell.R*2+1,1))
        #     fig = plt.figure()
        #     ax = fig.add_subplot(111, projection='3d')
        #     ax.plot_surface(X, Y, smooth_mean, cmap='viridis')
        #     ax.set_title('Dwelling B0 sampled from distribution \n(by integer location; smoothed gaussian filter, sigma={})'.format(sig))
        #     ax.set_xlim([-1,dwell.R*2+1])
        #     ax.set_ylim([-1,dwell.R*2+1])
        #     ax.invert_xaxis()
        #     plt.pause(0.1)    

        for x in range(len(B0)):
            B0[x] = smooth_mean[int(np.round(dr_coord[x,0])*inc)][int(np.round(dr_coord[x,1])*inc)]
        return B0

    def B0_sample(self, num):
        """ 
        ***NOT USED***
        -- included as example B0 setting method.

        B0_sample(num) samples dwelling prices B0 from range [0.001, 0.999] using the existing 
        income distribution calcs.F_s as the probability distribution.
        
        param num: number of dwellings to generate B0 for.
        """

        B0_scale = np.linspace(1, 0.1, 999)     # example B0 range

        # sample B0 from income distribution calcs.F_s
        B0_draw = np.random.choice(B0_scale, num, p=calcs.F_s)   
        B0 = np.array([B0_draw], dtype=float).T
        
        return B0

    def coordinates(self, num):
        """
        coordinates(num) generates unique coordinates for 'num' number of dwellings.
        This method also compares to the existing dwelling locations to ensure uniqueness.

        param num: number of additional coordinates generate
        return: concatenated coordinate vector returned, size 2*num

        """
        print('generating unique dwelling locations')

        dims = 2    # spatial dimensions; just use 2 as (X,Y) for now. 3 will include apartments but need to do ensure they're 'stacked' into a tower accordingly
        if np.sum(self.r_coord) == 0:       # check if new or adding to existing
            d_r_coord = np.round(np.random.random_sample(size=(self.M, dims))*self.R, 1)    # generate random 2D set of coordinates, 
            size_chk = num
        else:           # add to existing
            d_r_coord = np.round(np.random.random_sample(size=(num, dims))*self.R, 1)    # generate random 2D set of coordinates, 
            d_r_coord = np.concatenate((self.r_coord, d_r_coord))
            size_chk = self.M + num

        """ to insert custom coordinates of dwellings, add them here to specified index """
        # d_r_coord[0] = [0, 0]     # example insertion of coords (2D)

        # check for duplicates of coordinates then resample duplicates
        uniq, ind  = np.unique(d_r_coord, axis=0, return_index=True) 
        unsorted = np.array(sorted(np.concatenate((np.array(uniq), np.array([ind]).T), axis=1), key=lambda x:x[2]))
        uniq = np.delete(unsorted, -1, axis=1)

        while True:
            if len(uniq) == size_chk:
                r_coord = uniq 
                break 
            else:
                d_r_coord = np.round(np.random.random_sample(size=(size_chk-len(uniq), dims))*self.R, 1)    # generate random 2D set of coordinates, 

                uniq, ind = np.unique(np.concatenate((uniq, d_r_coord)), axis=0, return_index=True)
                unsorted = np.array(sorted(np.concatenate((np.array(uniq), np.array([ind]).T), axis=1), key=lambda x:x[2]))
                uniq = np.delete(unsorted, -1, axis=1)

        r_coord = r_coord[-num:]
        return r_coord

    def nearby_coords(self, nearby, num):
        """
        nearby_coords(nearby) generates unique coordinates for 'num' number of dwellings,
        at location nearby to previously transacted dwellings.

        This method also compares to the existing dwelling locations to ensure uniqueness.

        - effectively same function as 'coordinates()' but operates via distance to specific dwellings.

        param num: number of additional coordinates generate
        param nearby: nearby dwellings to find similar location
        return: concatenated coordinate vector returned, size 2*num
        """

        dims = 2    # spatial dimensions; just use 2 as (X,Y) for now. 3 will include apartments but need to do ensure they're 'stacked' into a tower accordingly
        d_r_coord = np.round(np.random.random_sample(size=(num, dims))*self.R, 1)    # generate random 2D set of coordinates, 
        d_r_coord = np.concatenate((self.r_coord, d_r_coord))
        size_chk = self.M + num

        # check for duplicates of coordinates then resample duplicates
        uniq, ind  = np.unique(d_r_coord, axis=0, return_index=True) 
        unsorted = np.array(sorted(np.concatenate((np.array(uniq), np.array([ind]).T), axis=1), key=lambda x:x[2]))
        uniq = np.delete(unsorted, -1, axis=1)

        exist_r_coord = self.r_coord[nearby]    # existing coordinates
        tmp = uniq[len(self.r_coord):]
        dwell_1 = np.reshape(np.repeat(exist_r_coord, len(tmp), axis=1).T, (2,-1)).T
        dwell_2 = np.reshape(np.repeat(tmp, len(exist_r_coord), axis=0).T, (2,-1)).T
        euclids = np.reshape(np.linalg.norm(dwell_1.T - dwell_2.T, axis=0), (len(exist_r_coord), -1)).T

        tmp = uniq[len(self.r_coord):]
        uniq_tmp = tmp[np.unique(np.where(euclids<=sim.h*3)[0])]
        uniq = np.concatenate((np.array(self.r_coord), np.array(uniq_tmp)),axis=0)

        while True:
            if len(uniq) == size_chk:
                r_coord = uniq 
                break 
            else:
                d_r_coord = np.round(np.random.random_sample(size=(size_chk-len(uniq), dims))*self.R, 1)    # generate random 2D set of coordinates, 

                uniq, ind = np.unique(np.concatenate((uniq, d_r_coord)), axis=0, return_index=True)
                unsorted = np.array(sorted(np.concatenate((np.array(uniq), np.array([ind]).T), axis=1), key=lambda x:x[2]))
                uniq = np.delete(unsorted, -1, axis=1)

                # chk distance
                tmp = uniq[len(self.r_coord):]
                dwell_1 = np.reshape(np.repeat(exist_r_coord, len(tmp), axis=1).T, (2,-1)).T
                dwell_2 = np.reshape(np.repeat(tmp, len(exist_r_coord), axis=0).T, (2,-1)).T
                euclids = np.reshape(np.linalg.norm(dwell_1.T - dwell_2.T, axis=0), (len(exist_r_coord), -1)).T

                tmp = uniq[len(self.r_coord):]
                uniq_tmp = tmp[np.unique(np.where(euclids<=sim.h*3)[0])]
                uniq = np.concatenate((np.array(self.r_coord), np.array(uniq_tmp)),axis=0)

        r_coord = r_coord[-num:]
        return r_coord

    def general(self, num):
        """
        general(num) generates general dwelling information [dwelling ID, on/off-market flag]

        param num: length of general info vector to generate
        return: concatenated general info vector returned, size 2*num
        """

        r_0 = np.linspace(0, num-1, num, dtype=int)      # dwelling ID (trivial)
        r_1 = np.ones(shape=(num, 1), dtype=int)    # on-market flag of dwelling
        r = np.concatenate((np.array([r_0]).T, r_1), axis=1)   # compile all general dwelling information

        return r

    def utility_0_new(self, r_coord):
        """ 
        utility_0_new(r_coord) samples dU_0 from the U_0 distribution of existing dwellings 
        
        param r_coord: input coordinates to sample from the existing U_0 mapping
        return: output u_0 vector of input size(r_coord)
        """

        r_coord_ref = np.array(np.round(r_coord * 4), dtype=int)
        if np.size(self.centric) == 0:
            u_0 = np.zeros((np.shape(r_coord)[0],1), dtype=object) * self.price_scale
        else:
            u_0 = np.ones((np.shape(r_coord)[0],1), dtype=object)# * self.price_scale

        # sample from the generated U0 distribution
        for x in range(len(r_coord)):
            u_0[x] = u_0[x] * self.U0_store[r_coord_ref[x,0]][r_coord_ref[x,1]] * self.price_scale
        
        return u_0

    def dwell_shuffle(self):
        """ 
        ***NOT USED***
        -- included as example of shuffling characteristic by another property.

        dwell_shuffle() shuffles the number of bedrooms of a dwelling as according to the U0 value so as to represent reality where 
        inner-city is full of apartments/smaller dwellings and the size generally increases with distance from the center of the city.

        """

        # U0 is chosen as a filter/means for sorting as it is a measure for distance from a central point.
        u0 = np.array(sorted(enumerate(self.r[:,2]), key=lambda x:x[1]))[:,0]
        B2_u0 = np.concatenate((np.array([sorted(self.B[:,2])]), np.array([u0])), axis=0).T
        B2_u0 = np.array(sorted(B2_u0, key=lambda x:x[1])).T
        self.B[:,2] = B2_u0[0]

    def B1_gen(self):
        """ 
        *** ONLY USED IN CASE C ***
        B1_gen() generates B_1(r) values for dwelling characteristic vector 

        atm only has capacity for 2 centres, needs to be refactored.
        """

        b1_max = self.price_scale*4

        self.b1_cent_1 = np.array([[0.7, 0.7]]) * sim.R
        self.b1_cent_2 = np.array([]) * sim.R
        if len(self.b1_cent_2) ==0:
            qq = np.array([self.b1_cent_1])
        else:
            qq = np.array([self.b1_cent_1, self.b1_cent_2])

        self.incs = self.R*4 + 1
        dim = np.linspace(0, self.R, self.incs)
        x_map, y_map = np.meshgrid(dim, dim)
        u0_map = np.concatenate((np.array([x_map.flatten()]).T, np.array([y_map.flatten()]).T), axis=1)
        
        if np.size(self.b1_cent_1) > 0:
            B1_1_dist = []
            for x in range(len(self.b1_cent_1)):
                B1_1_dist.append(np.linalg.norm(u0_map.T - np.array([self.b1_cent_1[x]]).T, axis=0))

            x = np.min(B1_1_dist, axis=0)
            B1_1_str = b1_max*np.exp(-abs(x**2)/(4**2))
        else:
            B1_1_str = []

        if np.size(self.b1_cent_2) > 0:
            B1_2_dist = []
            for x in range(len(self.b1_cent_2)):
                B1_2_dist.append(np.linalg.norm(u0_map.T - np.array([self.b1_cent_2[x]]).T, axis=0))

            x = np.min(B1_2_dist, axis=0)
            B1_2_str = -b1_max*np.exp(-abs(x**2)/(4**2))
        else:
            B1_2_str = []

        # scale to +- b1_max limits
        if np.size(self.b1_cent_1)>0 and np.size(self.b1_cent_2)>0:
            s1_str = B1_1_str + B1_2_str
        elif np.size(self.b1_cent_1)>0 and np.size(self.b1_cent_2)==0:
            s1_str = B1_1_str
        elif np.size(self.b1_cent_1)==0 and np.size(self.b1_cent_2)>0:
            s1_str = B1_2_str

        s1_str += abs(np.min(s1_str))
        s1_str /= np.max(s1_str)
        s1_str *= b1_max*2
        s1_str -= b1_max
        s1_str = np.reshape(s1_str, (self.incs,-1))
        self.B1_dist_val = np.round(s1_str,3)

        r_coord_ref = np.array(np.round(self.r_coord * 4), dtype=int)

        B_1 = np.zeros(shape=(self.M,1))

        # assign B1 to dwellings
        for x in range(len(B_1)):
                B_1[x] = self.B1_dist_val[r_coord_ref[x,0]][r_coord_ref[x,1]]

        self.B = np.round(np.array(np.concatenate((self.B, B_1), axis=1),dtype=float),3)

    def perturb(self):  
        """ 
        perturb() alters characteristics of dwellings; age is only characteristic/property that is altered for now.
        
        Future I: could also adapt the characteristics of the occupied dwellings to that of the occupying household,
            possibly at a cost to the household (mimicking renovation/extensions, requires investment model).

        """

        self.r[np.where(self.r[:,1]!=0), 1] += 1    # increase age/period of uninhabitance 

        # dwell.update_unocc_local_B0()
        
    def update_unocc_local_B0(self):
        """ 
        ***NOT USED***
        -- included as example B0 update method.

        update_unocc_local_B0() updates the dwelling price B0 of dwellings that have been unoccupied for a period of time
        to the local max B0 of currently/recently occupied dwellings.

        'local' dwellings are  is determined via nearest 0.5km^2 coordinate grid then smoothed via gaussian filtering with
        sigma=2.

        """

        age = 4
        inc = 1
        sz = dwell.R*inc + 1

        # check for dwelling unoccupied for > 1.5 yrs or >6 timesteps
        un_occ = np.where(self.r[:,1]>age)

        if np.size(un_occ) !=0: 
            locs = np.zeros(shape=(sz,sz))
            w = np.zeros(shape=(sz,sz))

            for x in np.where(self.r[:,1]<=age)[0]:
                locs[int(np.round(dwell.r_coord[x,0])*inc)][int(np.round(dwell.r_coord[x,1])*inc)] += dwell.B[x,0]
                w[int(np.round(dwell.r_coord[x,0])*inc)][int(np.round(dwell.r_coord[x,1])*inc)] += 1

            locs /= w
            locs[np.isnan(locs)] = 0
            mean_grid = np.array(locs)

            sig = 1
            smooth_mean = gaussian_filter(mean_grid, sigma=sig)
            smooth_mean /= np.max(smooth_mean)
            smooth_mean *= np.max(dwell.B[np.where(self.r[:,1]<=age),0])

            # if sim.step%4==0:
            #     X,Y = np.meshgrid(np.arange(0,dwell.R*2+1,1),np.arange(0,dwell.R*2+1,1))
            #     fig = plt.figure()
            #     ax = fig.add_subplot(111, projection='3d')
            #     ax.plot_surface(X, Y, smooth_mean, cmap='viridis')
            #     ax.set_title('Dwelling B0 sampled from distribution \n(by integer location; smoothed gaussian filter, sigma={})'.format(sig))
            #     ax.set_xlim([-1,dwell.R*2+1])
            #     ax.set_ylim([-1,dwell.R*2+1])
            #     ax.invert_xaxis()
            #     plt.pause(0.1)

            for x in un_occ[0]:
                self.B[un_occ,0] = smooth_mean[int(np.round(dwell.r_coord[x,0])*inc)][int(np.round(dwell.r_coord[x,1])*inc)]

    def evolve(self):
        """ 
        evolve() contains methods relating to manipulation of the total number of dwellings; evolution of supply/demand.
            i.e. changing supply via increasing/decreasing the number of dwellings.
        The method for increasing the number of dwellings calls functions that are called in the construct() function
            during initilisation of dwellings, creates 'new' number of dwellings.
        The method for creasing deletes an 'old' number of random dwellings that have been unoccupied for >n periods,
            alternatively if purely random then any occupants will be set in-market.
        
        Both methods re-adjust the dwelling IDs to correspond with the total number of dwellings in the model.

        """

        """ DWELLING CREATION """
        new = int(sim.beta*self.M)    # no. dwellings to create
        if self.M + new <= self.M_max:
            pass
        else:
            new = self.M_max - self.M

        if new > 0 and sim.step/4 > 21:
            if hasattr(sim, 'new_min') == False:
                sim.new_min = sim.step  # timestep at which household creation begins
            dwell.growth(new)

        elif sim.beta!=0 and new == 0 and sim.step/4>21 and hasattr(sim, 'new_max') == False:
            sim.new_max = sim.step  # timestep at which household creation ceases

        """ DWELLING DESTRUCTION """
        if sim.step/4 > -1:
            demo = int(sim.lamb*self.M)      # no. dwellings to delete
            old = np.where(self.r[:,1]>=8)[0]   # delete dwellings off-market for more than 8 periods
            if demo > 0 and np.size(old)>0 and self.M > self.M_min*pop.N:
                dwell.destroy(demo, old)

    def growth(self, new):
        """
        growth() generates new dwellings to add to existing dwellings. Characteristics and all other info
        is generated as per the construct() function.
        New dwellings flagged as on-market.

        In present version, new dwellings take on random characteristics & random location.
        Paper refers to eq. 19 as method for generating new dwellings. 

        param new: input number of dwellings to create

        """

        """ CREATE DWELLING BY EQ. 19 """ 
        # duplicate recently transacted dwellings
        r = dwell.r
        # r_rec = np.where((r[:,1]!=0)&(r[:,1]<=4))[0]
        # r_rec = np.where(r[:,1]==0)[0]
        if len(self.r_recs)>2:
            self.r_recs = np.array(self.r_recs[1:])
        self.r_recs = np.concatenate((np.array(self.r_recs),np.array(calcs.rec_tran)),axis=0)

        # print(self.r_recs)
        r_rec = np.reshape(np.array(self.r_recs, dtype=int),(1,-1))[0]

        # self.r_recs.append(np.array(calcs.rec_tran))

        # generate coordinates nearby recently transacted
        dr_coord = dwell.nearby_coords(r_rec, new)

        # sample general utility from the existing distribution
        du_0 = dwell.utility_0_new(dr_coord)

        # duplicate dwelling characteristics
        dB = dwell.B[r_rec]

        """ ============================= """


        """ CREATE RANDOM DWELLING METHOD """
        # # sample new coords; cross-checks with existing coords for uniqueness but only returns fresh coordinates of size new
        # dr_coord = dwell.coordinates(new)

        # # sample general utility from the existing distribution
        # du_0 = dwell.utility_0_new(dr_coord) 

        # # generate new dwelling characteristics
        # dB = dwell.characteristics(new)
        """ ********************** """

        # this if statement probably isn't too necessary; just to keep dimensions in line for B = [B0] or =[B0, B1...]
        if np.shape(self.B)[1] == 1:
            dB = np.round(np.array([dB[:,0]], dtype=float).T *self.price_scale, 3)
        else:
            dB = np.round(np.array(dB*self.price_scale, dtype=float), 3)

        dr_ = dwell.general(new)
        dr_[:,0] += np.shape(self.r)[0]
        dr = np.concatenate((dr_, du_0), axis=1)   # compile general information of dwelling [ID, on-market flag, U_0]

        # scatter of new dwellings
        # plt.figure(tight_layout=True)
        # plt.scatter(self.r_coord[:,0],self.r_coord[:,1],s=10,marker='.',c='lightgrey')
        # X_p, Y_p = sim.r[:,1,0:2][np.array(sim.p[:,0,2], dtype=int)].T   # coordinates of each household
        # plt.scatter(X_p,Y_p,s=10,marker='.',c='black')
        # plt.scatter(dr_coord[:,0],dr_coord[:,1],s=10,marker='.',c='red')
        # plt.gca().set_aspect('equal', adjustable='box')
        # plt.show()

        # concatenate all new dwellings with existing
        self.u_0 = np.concatenate((self.u_0, du_0), axis=0)
        self.B = np.concatenate((self.B, dB), axis=0)
        self.r = np.concatenate((self.r, dr), axis=0)
        self.r_coord = np.concatenate((self.r_coord, dr_coord), axis=0)

        # re-assign total number of dwellings to dwell.M 
        self.M = len(self.u_0)

    def destroy(self, demo, old):
        """
        destroy() removes a number of dwellings from the model, setting previously occupying households in-market.
        Current selection method is random for dwellings uninhabited for period of time.  
        
        future I: could use negative skewed normal for the probability of demolition as function of dwelling 
        price so that houses of lower-middle will be demolished more often. This class of people usually inhabits
        dwellings that are below the global mean utility, also move/relocate more often.

        future II: could demolish occupied dwellings, then set these households in-market.

        param demo: input number of dwellings to remove

        """

        # index of dwellings to delete
        del_ind = np.random.choice(old, size=(demo,1), replace=False)    # dwellings off-market for 3*(perturbation step size) periods

        # set all households occupying these dwellings to on-market and their utilities to 0
            # setting U = 0 allows for them to take whatever is available when transacting 
            # so as to not allow for homelessness.
        onmark = np.where(pop.p[:,2] == del_ind)[1]
        pop.U[onmark] = 0       # set evicted household utility to null

        pop.p[onmark, 1] = 1    # set evicted household to on-market
        pop.p[onmark, 2] = -1   # set occupied dwelling to -1, this is necessary so that in ∆U calculations, 
                                # those with -1 household are assigned to the least worst dwelling

        # delete all indices of dwellings to be destroyed
        self.u_0 = np.delete(self.u_0, del_ind, axis=0)
        self.B = np.delete(self.B, del_ind, axis=0)
        self.r = np.delete(self.r, del_ind, axis=0)
        self.r_coord = np.delete(self.r_coord, del_ind, axis=0)

        # re-number dwellings & re-assign these dwelling numbers to corresponding households 
        old_r = np.array([self.r[:,0]]).T   # current dwelling IDs 
        new_r = np.array([np.linspace(0, len(self.r)-1, len(self.r), dtype=int)]).T    # new dwelling IDs (0-len(self.r))
        r_p_r = np.concatenate((old_r, new_r), axis=1)  # dwelling ID correspondence (old-new)
        ### FINE TO HERE (CORRESPONDENCE MATRIX IS FINE)
        housed = np.where(pop.p[:,2]!=-1)[0]    # households who occupy a dwelling after destruction
        old_r_loc = pop.p[:,2][housed]  # dwellings still occupied after destruction
        new_r_loc = np.where(r_p_r[:,0]==np.array([old_r_loc]).T)[1]   # these are the locations of the old dwelling IDs in the dwelling ID correspondence

        pop.p[housed,2] = r_p_r[new_r_loc,1]
        self.r[:,0] = r_p_r[:,1]

        # re-assign total number of dwellings to dwell.M 
        self.M = len(self.u_0)

class simulate:
    def __init__(self, case):
        """
        __init__() instantiates whole model: saving directory of model, calculation class, visualation class, household class, dwelling class.
        
        Optionality to generate data from scratch or load from pre-existing data.

        """
        self.case = case    # simulation mode

        self.sim_id = str(simtime.current())    # ID of current simulation
        self.current_folder = '{}-{}-{}'.format(self.sim_id[4:6],self.sim_id[2:4], self.sim_id[:2])
        self.sim = 'market_dyn'
        self.all_store_pop = []
        self.all_store_dwell = []

        # create a directory for simulation output
        chk_dir = os.path.isdir(str(diroutput) + self.current_folder)
        if chk_dir == False:
            os.mkdir(str(diroutput) + self.current_folder)
        self.sim_folder = '{}#{}'.format(self.sim, self.sim_id)
        os.mkdir(str(diroutput)+self.current_folder+'/'+self.sim_folder)
        self.save_loc = str(diroutput) + str(self.current_folder) + '/' + str(self.sim_folder) + '/' # save directory

        simulate.model_parameters(self) # instantiate the model parameters for simulation

    def model_parameters(self):
        """ 
        model_parameters() instantiates the main parameters in the model.

        """   

        """ DEFAULT VALUES """
        # General parameters for model
        self.R = 25                         # size of coordinate space 
        self.t = 25                         # no. years for simulation
        self.t_inc = 4                      # increments per year; annualized rates will be divided by this 
        self.chi = 1                        # probability of transaction clearance (decimal)
        self.sigma = 0.06/self.t_inc        # fractional move rate (evaluated t_inc/year)

        # Parameters for population and dwelling increase/decrease (evaluated 1/year)
        # rates are as per evaluation period i.e. per year/no. steps per year
        self.alpha = 0.04/self.t_inc       # rate of change of household preferences
        self.eps = 0.00/self.t_inc         # rate of population growth 
        self.rho = 0.00/self.t_inc         # rate of population decline
        self.beta = 0.00/self.t_inc        # rate of dwelling creation
        self.lamb = 0.00/self.t_inc        # rate of dwelling destruction

        # affinity function G
        self.G_0 = 0.2
        self.h = 1

        # income distribution parameters
        self.m_0 = 2.5e5            # maximal m in the population, $250,000 y^-1 in paper
        self.m_c = 5e4              # point beyond which the distribution falls off rapidly, $50,000 y^-1 in paper
        self.a = 2                  # income exponent

        # household parameters
        self.N = 1000               # number of households
        self.N_min = 1000           # absolute minimum number of households
        self.N0 = self.N            # original number of households -- used in plotting of mean B and mean U

        # dwelling parameters
        self.M = 3*self.N           # number of dwellings
        self.M_min = 1.2            # min no. dwellings M permitted (during dwelling destruction)
        self.M_max = 3*self.N       # max no. dwellings M permitted (during dwelling creation)
        self.u0_max = 5             # general dwelling utility U0
        self.u0_wid = 7             # characteristic width of U0 window for preferred area
        self.price_scale = 1000     # rough scale of $/quarter for pricing methods of dwellings.
        self.cent = []              # location of U0 peak centricity

        # general settings
        save_vectors = str(input('Do you wish to save household/dwelling vectors at each timstep? (yes/no)'))
        if save_vectors == 'yes':
            self.save_vectors = 1
        else:
            self.save_vectors = 0
        

        """ update parameters by case """
        # store of parameters to update
        case_param_dict = {'A': {'lamb':0.02/self.t_inc},
                'A50':{'lamb':0.02/self.t_inc, 'R':50, 'N':4000, 'M':12000},
                'B_slw': {'sigma':0.02/self.t_inc,'lamb':0.01/self.t_inc, 'cent':[[0.5, 0.5]]}, 
                'B': {'sigma':0.06/self.t_inc,'lamb':0.02/self.t_inc, 'cent':[[0.5, 0.5]]}, 
                'C': {'lamb':0.02/self.t_inc},
                'D': {'lamb':0.02/self.t_inc, 'cent':[[0.5, 0.5]]},
                'D_lrg': {'lamb':0.02/self.t_inc, 'R':50},
                'D_poly': {'lamb':0.02/self.t_inc, 'R':25, 'cent':[[0.2, 0.2],[0.8,0.8]]},
                'E_popinc': {'t':100, 'eps':0.03/self.t_inc, 'cent':[[0.5, 0.5]]},
                'E_popdec': {'t':100, 'sigma':0.07/self.t_inc,'rho':0.05/self.t_inc,'N':1500, 'N_min':300, 'M':2000, 'cent':[[0.5, 0.5]]},
                'E_dwellinc': {'t':100, 'beta':0.04/self.t_inc, 'M':1.5*self.N, 'M_max':3*self.N, 'cent':[[0.5, 0.5]]},
                'E_dwelldec': {'t':100, 'lamb':0.03/self.t_inc, 'cent':[[0.5, 0.5]]}}

        # update parameters for the case
        for key, val in case_param_dict[self.case].items():
            exec('self.'+key + '=val')
            print("{} = {}".format(key, val))

        self.full_case = str(self.case)
        self.case = str(self.case)[0]

    def create(self):
        """ 
        create() instantiates all objects/classes within the model: 
        households, dwellings, calculations, visualisations.

        """

        global calcs, pop, dwell, visu      # instantiate all objects within the model
        calcs = calculations()
        pop = household(self.N, self.N_min)
        dwell = dwelling(self.M, self.R, self.u0_max, self.u0_wid, self.price_scale, self.M_min, self.M_max, self.cent)

        clean_start = 1
        # while True:
        #     clean_start = int(input('Evolve from scratch (1) or from existing data (2):'))
        #     if clean_start == 1:
        #         # proceed as normal & fill all profile arrays
        #         break
        #     elif clean_start == 2:
        #         # load the saved data to begin simulation, only the parameters for length of simulation will remain, 
        #         # all parameters generated during instantiation of objects will be overwritten with the saved parameters.
        #         simulate.load_from_save(self)
        #         break
        #     else:
        #         break 
        #         print('Invalid mode entered.')
        
        if clean_start == 1:
            pop.populate()      # generate households
            dwell.construct()   # generate dwellings
            pop.assign_dwellings()  # assign all households to unique dwellings
            print('MODEL INITIALISED')

        """ MODEL SETUP VISUALISATION """
        visu = visualisation(sim, pop, dwell, calcs)
        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['font.family'] = 'STIXGeneral'
        # U0 distribution
        if sim.case =='B':
            if hasattr(dwell, 'U0_store'):
                visu.U0_dist_2d()        # U0 distribution generated (2d plot as function of r) - figure in paper
                # visu.U0_dist_3d()        # U0 distribution generated (3d plot)
        # visu.U0_sample_mean()        # mean U0 distribution from sample
        # visu.U0_sample()             # U0 distribution from sample
        # income distribution
        # visu.figure_4b_normalised()   # normalized version of figure 4b from research notes
        # visu.m_s0_p()           # income m(p) and price sensitivity s_0(p) vs household p (double axis)
        # visu.s0_N()               # scatter plot of the distribution of s_0 by household
        # case C - B1 distribution
        elif sim.case == 'C':
            visu.B1_dist()          # figure 11a distribution of B1 values in dwellings

        """ Figures 2-6 from paper """
        # visu.figure_2()     # figure 2a,b from paper
        # visu.figure_3()     # figure 3 from paper
        # visu.figure_4abc()  # figure 4a,b,c from paper    
        # visu.figure_5()     # figure 5 from paper

        plt.show()
        # exit()

        start = input("""Press return/enter to begin simulation with above parameters. (any key + enter to exit)""")
        if len(start)!=0:
            exit()

    def load_from_save(self):
        """
        load_from_save() loads all saved data from a specified save folder and fills this data
        into all profile arrays.

        """

        load_id = int(input('Enter the ID of the saved simulation to load: (number following #)'))
        print("This component hasn't been built yet, come back in 2032")
        exit()

        ## use these lines below
        # load_files = sorted([filenames for _,_,filenames in os.walk(sim.save_loc)][0])
        # print(load_files)

        pass

    def evolve(self):  
        """ 
        evolve() is the core evaluation component of the model. 
        Iterates through time then handles all output & plotting at the end.

        """

        #iterate over n timesteps, all population & housing evolution rates will need to be set for quartely iterations.
        for step in range(self.t*self.t_inc):  
            print("STEP NO. {} of {} ".format(step+1, self.t*self.t_inc))
            sim.step = step

            # update household utility and dwelling price B0
            calcs.U_B0_update()

            # save model data at each step
            sim.saving(step)

            # 2. population evolution 
            pop.evolve()

            # 3. evolve the supply of housing, demolition & construction,
            dwell.evolve()

            if self.sigma > 0:
                # 1. Calculate a 2dimensional utility matrix for all on-market households for all dwellings.
                calcs.utility_house_dwell()

                # 4. carry out transactions 
                calcs.transactions()

            # 5. evolve preferences of households and characteristics of dwellings
            pop.perturb()
            dwell.perturb()

        # final save after simulation complete
        calcs.U_B0_update()
        sim.saving(step)
        print('t2f {}'.format(time.time() - start))

        """ RESULTS VISUALISATION """
        visu = visualisation(sim, pop, dwell, calcs)
        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['font.family'] = 'STIXGeneral'

        if sim.case == 'A':
            # City Formation
            visu.spatial_gaussian()     # figure 7a-f
            # if sim.full_case == 'A50':  
                # visu.spatial_gaussian_movie()     # 7g .gif creation

        elif sim.case == 'B':
            # Effect of U0
            # figure 8,9: use seed 69, higher transaction/evolution rates also.
            visu.densitymap_whole()     # figure 8b
            visu.densitymap_wealthlevel()   # figure 9a-f
            visu.meanB_r_10()           # figure 10a: use seed 4, lamb=0.01, sigma=0.02
            visu.U_contribution()       # figure 10b,c: use seed 69, lamb=0.01, sigma=0.02

        elif sim.case == 'C':
            # Effects of Spatial Localization of Favored Dwelling Characteristics
            visu.C_gaussian_map()   # figure 11b,c: gaussian smoothened spatial density of households
            visu.C_heatmap()        # greyscale heatmap of spatial density of households

        elif sim.case == 'D':
            # Segregation
            visu.spatial_types()    # figure 12,13: spatial plot of households by characteristic type q1
            visu.gaussian_types()   # gaussian filter of density of households by characteristic type q1
            # if sim.full_case == 'D_lrg':
                # visu.spatial_types_movie()  # figure 13 as gif: spatial plot of households by characteristic type q1
        
        elif sim.case == 'E':
            # Supply and Demand
            if sim.full_case == 'E_dwellinc' or sim.full_case == 'E_dwelldec':
                visu.M_t()              # no. dwellings M vs time t
            elif sim.full_case == 'E_popinc' or sim.full_case == 'E_popdec':
                visu.N_t()              # figure 14a: no. households N vs time t
            visu.meanB_t_r()        # figure 14b: plot mean B at specific radiuses for all t
            visu.meanB_r_t()        # figure 14c: plot of mean B at all radiuses for each t
            visu.mean_density()     # figure 14d: plot of mean density at all radius for each t  
            visu.spatial_dist()

        elif sim.case == '9':
            # custom mode
            visu.spatial_dist()

        plt.show()
        exit()

        # sim.p = np.array(sorted(sim.p, key=lambda x:x[2][0]))   # sort by wealth

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

        # visu.liveplot()               # animated plot of the spatial distribution
        # visu.interactive_spatial()    # interactive plot of spatial distribution of households/dwellings

        exit()

        # save the model.py file (as backup and also as means to re-run previous models)
        for x in ['market_dynamics.py']:
            shutil.copyfile(dirmain +'/'+ x, self.save_loc + x)

    def saving(self, timestep):
        """
        saving(timestep) saves all household and dwelling vectors as separate files within specific simulation-ID/datetime folder of iteration,
            i.e. households_timestep-n.txt, _utilities_timestep-n.txt, dwellings_timestep-n.txt
        The exact 'market_dynamics.py' file is saved also.

        param timestep: 

        """
        # save all vectors related to the household and the dwelling as separate files within specific datetime folder of iteration. 
        # save all model parameters also; model parameter file will include runtime also.   -- NEED TO ADD THIS IN
        # These will be saved in a format such that it can be input as starting data

        print('SAVING DATA')

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

        ## save household profile vector -- this is just turned off for testing
        if self.save_vectors == 1:
            with open(self.save_loc +'households_' + self.sim_id + '_timestep-{}.txt'.format(timestep) , 'w') as save:
                save.write('# household profile vector shape (no. households, 4, no. preferences/max length)\n')
                np.savetxt(save, p_dim, fmt='%-7.2f')
                save.write("""# KEY:
    # [[Household ID, on-market flag, dwelling_id, utility, 0...],
    # [characteristics: q_0(p), q_1(p), q_2(p), ..., q_m(p)],
    # [housing preferences: [s_0(p), s'(p)] = [s_0(p), s'_1(p), s'_2(p), ..., s_n(p)] ]\n""")

                for data_slice in self.p:
                    save.write('\n# household {} \n'.format(data_slice[0][0]))
                    np.savetxt(save, data_slice, fmt='%-7.2f')
        
        """ compile the dwelling profile array for saving """
        r_mn = max([len(dwell.style), len(pop.s_bound)])    # max of length of characteristics B or length of household profile array
        self.r = np.zeros((dwell.M, 3, r_mn), dtype=object)   # initialise the empty array dimensions to save dwelling data
        r_dim = np.array(np.shape(self.r))
        r_dim = np.array([np.pad(r_dim, (0,r_mn-len(r_dim)), 'constant')])

        for n, [r, r_coord, B] in enumerate(zip(dwell.r[0:], dwell.r_coord[0:], dwell.B[0:])):
            self.r[n][0][:len(r)]       = np.array(r)        # assign general information of dwelling
            self.r[n][1][:len(r_coord)] = np.array(r_coord)    # assign dwelling location
            self.r[n][2][:len(B)]       = np.array(B)        # assign dwelling characteristics
            
        ## save dwelling profile vector -- this is just turned off for testing
        if self.save_vectors == 1:
            with open(self.save_loc +'dwellings_' + self.sim_id + '_timestep-{}.txt'.format(timestep) , 'w') as save:
                save.write('# dwellings profile vector shape (no. dwellings, 3, no. characteristics/max length)\n')
                np.savetxt(save, r_dim, fmt='%-7.2f')
                save.write("""# KEY:
    # [[Dwelling ID, on-market flag, U_0(r), 0, ...],
    # [location: r_1(r), r_2(r), 0, 0, 0, 0, ...],
    # [characteristics: B_0(r), B_1(r), ..., B_n(r)]]\n""")

                for data_slice in self.r:
                    save.write('\n# dwelling {} \n'.format(data_slice[0][0]))
                    np.savetxt(save, data_slice, fmt='%-7.2f')

        self.all_store_pop.append(np.array(self.p, dtype=object))
        self.all_store_dwell.append(np.array(self.r, dtype=object))

        print('HOUSEHOLD & DWELLING PROFILE VECTORS SAVED')

class calculations:
    def __init__(self):
        self.fees = 0.00    # percentage of price of dwelling which will be charged as fees for transaction

    def sample_s0_income_dist(self, num):
        """
        sample_s0_income_dist(num) takes 'num' samples from a pre-defined income distribution. Primarily used in the creation of new households.

        param num: number of samples to take i.e. number of new households for which need an income from a realistic distribution

        """

        s_0 = np.linspace(-0.001, -0.999, 999)   # uniformly distributed, later re-sample s_0 from the income distribution
        draw = np.random.choice(s_0, num, p=self.F_s_norm)   # sample from a pre-defined income distribution self.F_s
        s_0 = np.array([draw], dtype=float).T

        return s_0

    def price_sens_income_dist(self):
        """
        price_sens_income_dist() generates an income distribution as defined by equation 15 in the paper as from a uniform range of price sensitivities, 
            then re-samples the price sensitivity as from the income distribution.
            
        return: price sensitivity s_0 as sampled from an income distribution given by equation 15 within the paper.

        """

        print('calculating s_0 from income distribution')
        self.s_0 = np.linspace(-0.001, -0.999, 999)   # uniformly distributed, later re-sample s_0 from the income distribution

        # income distribution from eq. 15
        m_0 = sim.m_0                 # maximal m in the population, $250,000 y^-1 in paper
        self.m = m_0*np.sqrt(1-self.s_0**2)   # income per unit time available for housing of p; eq. 11
        a = sim.a                       # income exponent
        self.m_c = sim.m_c                   # point beyond which the distribution falls off rapidly, $50,000 y^-1 in paper
        b = 0.01                    # normalization constant (arbitrary initial value before analytic convergence)
        
        # iteratively evaluate for the normalisation constant b
        area = 0
        z=1
        while True:
            self.F_m = b*(1 + self.m**2/self.m_c**2)**(-a/2)   # eq. 14, N = integral of eq. 14
            area = abs(np.trapz(y=self.F_m, x=self.m))
            if np.round(area) == pop.N:
                break
            else:
                if area<pop.N:
                    b += 0.01/z
                elif area > pop.N:
                    b -= 0.005/z
                    z += 1

        self.F_s = (b*m_0*abs(self.s_0))/(np.sqrt(1-self.s_0**2)*(1+(1-self.s_0**2)*m_0**2/self.m_c**2)**(a/2))  # eq. 15
        self.F_s_norm = self.F_s/sum(self.F_s)  # normalise

        self.draw = np.random.choice(self.s_0, pop.N, p=self.F_s_norm)
        self.s_0_sampled = np.sort([self.draw]).T
        self.m_sampled = m_0*np.sqrt(1-self.s_0**2)

        return np.array(self.s_0_sampled, dtype=float)

    def U_B0_update(self):
        """ 
        U_B0_update() updates the utility value for ALL households in their current dwelling, 
            so as to update their utility impact of other household movements around them.
            B0 then updated as according to the household's updated utility.
        Calculated on a reduced timeframe i.e. on yearly timeframe, 

        """

        # household data
        p = pop.p
        U = pop.U
        q = pop.q
        s = pop.s

        # set all occupied dwellings off-market
        dwell.r[p[:,2], 1] = 0

        # corresponding dwelling data for each household occupying it
        r = dwell.r[p[:,2]]
        U0 = r[:,2]
        r_coord = dwell.r_coord[p[:,2]]
        B = dwell.B[p[:,2]]

        U_new = U0 + np.sum(s*B, axis=1)    # updated utility for their current dwelling
        self.u0 = U0

        # calculate euclidean distance between all households
        loc = dwell.r_coord[p[:,2], 0:2]
        dwell_1 = np.reshape(np.repeat(loc, len(loc), axis=1).T, (2,-1)).T
        dwell_2 = np.reshape(np.repeat(loc, len(loc), axis=0).T, (2,-1)).T
        euclids = np.reshape(np.linalg.norm(dwell_1.T - dwell_2.T, axis=0), (len(loc), -1))

        # calculate final term of the utility equation for different types of households.
        s0 = s[:,0]
        self.s0 = s0
        q = np.array(pop.q)
        G = sim.G_0*np.exp(-euclids**2/sim.h**2)
        U_G = np.sum(np.dot(q, q.T)* G, axis=1)* dwell.price_scale 
        self.qq = U_G
        pop.U = np.round(np.array([U_new + U_G], dtype=float).T, 3)
        self.u = pop.U

        # dwelling identifier by household
        p = pop.p

        # udpate price of occupied dwellings - set B0 to U' = U - s0B0
        if np.shape(dwell.B[p[:,2]])[1] == 1:
            dB = pop.U - pop.s*dwell.B[p[:,2]]
        else:
            dB = pop.U - np.array([pop.s[:,0]*dwell.B[p[:,2], 0]]).T

        dwell.B[p[:,2],0] = np.round(np.array(dB.T, dtype=float), 3)
        self.sb = s0*dB.T
        self.sb = self.sb[0]    
        self.b0 = dB

    def utility_house_dwell(self):
        """ 
        utility_house_dwell() calculates the marginal utility for all in-market households for all on-market dwellings 
        to be used in decision to transact.

        This method calculates the s.B dot product then evaluates the final utility term iteratively as outlined below.
        Current method sorts the households via wealth decreasing such that the transactions occur via wealth.
        
        """

        # calculate current utility difference for on-market households for all dwellings.

        # P(r) = max U(r,p) of a dwelling is the utility/income stream required by the household to obtain the dwelling. 
        # calculate the dwelling/household utilities, iterate from s_0 =0 to s_0=-1, 
        # move all households whose utilities increase (and is maximised). Can also move them only if marginal utility is above a threshold?
        # assign household to that dwelling, set off-market flags for househld & dwelling, flag previous dwelling on-market
            
        #take a portion of households to transact at given timestep, set on-market 
        p_mark = np.random.choice(len(pop.p), size=(1, int(np.ceil(sim.sigma*len(pop.p)))), replace=False)[0]
        pop.p[:,1][p_mark] = 1              # set portion of households in-market

        p_mark = np.where(pop.p[:,1]==1)    # take all on-market households
        self.p_mark = np.array([x for _, x in sorted(zip(pop.s[p_mark], pop.p[p_mark]), \
            key=lambda x: x[0][0], reverse=True)])                      # on-market households, sort by wealth decreasing 

        self.r_mark = dwell.r[np.where(dwell.r[:,1]!=0)]                          # on-market dwellings
        self.p_r_utils = np.zeros((np.shape(self.r_mark)[0], \
            np.shape(self.p_mark)[0]))                                  # household-dwelling utility array
        self.s0b0 = np.zeros((np.shape(self.r_mark)[0], \
            np.shape(self.p_mark)[0]))          # price component of utility; to be subtracted later for dwelling price adjustment

        """ calculate the household-dwelling utility array """
        calcs.dotprod_sB()              # calculate U0 + s*B product
        calcs.utility_qq()              # calculate final utility term with characteristics q &
                                            # and add to household dwelling utility array
        calcs.estimate_B0_unocc()       # estimate B0 from U' for on-market dwellings

        self.U_c = np.array(pop.U[self.p_mark[:,0]], dtype=float)      # utility of current on-market households
        self.p_r_utils -= self.U_c.T    # subtract current utility for households to obtain marginal utility for each dwelling

    def estimate_B0_unocc(self):
        """ 
        estimate_B0_unocc() calculates the potential dwelling price B0 of all on-market dwellings
        for all in-market households via the U' of these dwellings.

        """

        """ estimate B0 from U' """
        B0_ = self.p_r_utils - self.s0b0             # obtain the U' estimate for B0                      
        # B0_ = np.array([np.max(B0_,axis=1)]).T              # B0 estimated from maxU' for each dwelling

        """ estimate B0 from U' bids in the local area. """
        B0_bar = np.zeros(shape=(np.shape(B0_)[0],1))
        inc = 1              # neighbourhood = 1km/inc = 0.5km
        sz = int(dwell.R*inc + 1)

        un_occ = self.r_mark[:,0]
        occ = np.where(dwell.r[:,1]==0)[0]

        if np.size(un_occ) != 0:
            locs = np.zeros(shape=(sz,sz))
            w = np.zeros(shape=(sz,sz))

            # for p_on, _ in enumerate(self.p_mark): 
            for r_on, x in enumerate(un_occ):
                locs[int(np.round(dwell.r_coord[x,0])*inc)][int(np.round(dwell.r_coord[x,1])*inc)] += np.mean(B0_[r_on, :])
                w[int(np.round(dwell.r_coord[x,0])*inc)][int(np.round(dwell.r_coord[x,1])*inc)] += 1

            locs /= w
            loc_mean = np.nanmean(locs)
            locs[np.isnan(locs)] = loc_mean
            locs[np.isinf(locs)] = loc_mean
            mean_grid = np.array(locs)

            sig = 1
            smooth_mean = gaussian_filter(mean_grid, sigma=sig)
            smooth_mean[np.isnan(smooth_mean)] = 0
            
            for y, x in enumerate(un_occ):
                B0_bar[y] = smooth_mean[int(np.round(dwell.r_coord[x,0])*inc)][int(np.round(dwell.r_coord[x,1])*inc)]    
            B0_bar = np.array(B0_bar)

        """ B0 weighted sum of contributions"""
        x = 0.5
        B0_og = B0_ * x     # B0 - individual bids
        B0_bar *= (1-x)     # B0 - local neighbourhood

        B0 = np.array(B0_og+B0_bar).T
        s0 = np.array([pop.s[self.p_mark[:,0], 0]]).T       # household price sensitivity
        upd_s0B0 = np.array(s0*B0).T
    
        self.p_r_utils = B0_ + upd_s0B0              # update p_r_utils accordingly with new B0 estimate
        self.s0b0 = upd_s0B0                    # update s0b0 store to subtract for U' in transactions

    def dotprod_sB(self):
        """ 
        dotprod_sB() calculates the dot product of household preference and dwelling characteristic vector 
        and adds U0 for all dwellings.

        row = dwelling from r_mark, column = households' (from p_mark) utility for specific dwelling
        """

        for y, r in enumerate(self.r_mark):
            U0 = r[2]   # general utility U0 of dwelling 
            B = dwell.B[r[0]]   # dwelling preference vector
            s = pop.s[self.p_mark[:,0]]     # preferences of households

            self.p_r_utils[y] = U0 + np.dot(s, B)  # general utility + (preference * characteristic)

            # store s0*B0 contribution
            if np.shape(s)[1] == 1:
                self.s0b0[y] = np.dot(s, B)
            else:
                self.s0b0[y] = s[:,0]*B[0]

    def utility_qq(self):
        """
        utility_qq() calculates last term of utility function involving household characteristics.
        
        Method:
        - For all households on-market, iterate throguh all dwellings on-market x times (x = no. household types), 
            and calculate the marginal utility as according to the locality of the considered dwelling to households 
            with similar/disimilar characteristics to their own.
        - In each repetititon, those of similar household types will have their q_1 value set to 1 to represent attraction, 
            and those with disimilar types will have their q_1 values set to 0 to represent no attraction. for repulsion, set disimilar to -1.        
        
        """

        rmark = np.array([self.r_mark[:,0]], dtype=int)[0]  # on-market dwellings
        dwell_on = dwell.r_coord[rmark, 0:2]                # coordinates of on-market dwellings
        off_mark = pop.p[pop.p[:,1]==0]                     # off-market households
        pop_off = dwell.r_coord[off_mark[:,2], 0:2]         # coordinates of off-market households

        # broadcasting (repeats for euclid distance calc),
        # then calculate euclid dist between off-market households and on-market dwellings
        pop_off_rep = np.reshape(np.repeat(pop_off.T, len(dwell_on), axis=1), (2,-1))
        dwell_on_rep = np.reshape(np.repeat(dwell_on.T, len(off_mark), axis=0), (2,-1))
        euclids = np.reshape(np.linalg.norm(pop_off_rep - dwell_on_rep, axis=0), (len(pop_off),-1)).T

        # calculate the final term of the utility function 
        q_off = pop.q[off_mark[:,0]]
        q_on = pop.q[self.p_mark[:,0]]
        G = sim.G_0*np.exp(-euclids**2/sim.h**2).T
        self.final_U = np.dot(np.dot(q_on, q_off.T), G).T* dwell.price_scale

        self.p_r_utils += self.final_U  # add final utility term to household dwelling utility array

    def transactions(self):
        """
        transactions() evaluates transactions for all in-market households for all on-market dwellings as according to their maximised marginal utility
            as calculated in the utility_house_dwell() function.

        Transaction execution process (from research notes):
        1. if a bidder is successful in transaction with probability chi, then they acquire the dwelling for 
            which their utility is maximised i.e. dwelling with the largest marginal utility ∆U.
            Otherwise if the household is unsuccessful in transaction with probability 1-chi, or there are no dwellings for which their ∆U >0,
            then they are removed from the bidding pool.
        2. household utility U drives market participation, and P_r = U', then the newly 'sold' dwelling has B0 = P_r = U'.
        3. transactions in 1) are completed and ∆U is recalculated without those market participants and dwellings, 
            in this code the dwelling sold and the household that has transacted is removed from the bidding pool.
        3. above steps repeated until all people are housed. -- the prior dwellings of those who have successfully transacted
            are not re-marketed, however a flag is attached for the dwelling to be labelled as unoccupied; only on-market dwellings
            at the beginning of the transaction process are available for bidding.

        Current method notes:
        P(r) is not recalculated as the marginal utility array is pre-calculated for all households and dwellings, 
            the relevant dwelling(s)/household(s) are instead removed from the 'household dwelling marginal utility' array defined as p_r_utils.
        Future work where investment is included could incorporate the sale price of their previous dwelling in a factor that alters a households'
        wealth and/or price sensitivity s0.
        """

        # random.choices(population, weights=None, *, cum_weights=None, k=1) for selecting order of transaction as according to Pareto dist as function of wealth
        self.rec_tran = []  # recent transactions

        # re-locate households to dwelling where their marginal utility is positive and maximal
        for y, p in enumerate(self.p_mark):
            x = 0   # column of household transacting; set to 0 for now but could introduce alternative ordering in future. 

            # if no dwellings on-market -> break
            if np.size(self.r_mark) == 0:
                break

            clear = np.random.choice([0, 1], 1, p=[1-sim.chi, sim.chi])     # probability chi of clearance; set to 1 for current work
            if clear == 1:                                                  # check transaction success
                util_max = self.p_r_utils[:,x].argmax()                 # index of dwelling with largest marginal utility
                util = self.p_r_utils[util_max,x]                       # value of largest marginal utility
                best = self.r_mark[util_max]                            # dwelling of maximal increase in utility

                if util > 0 or p[2] == -1:              # check if marginal utility for any dwelling > 0, 
                                                        # OR if occupied dwelling has been destroyed.
                    
                    if p[2] != -1:                      # set previous dwelling to on-market for next timestep 
                        dwell.r[p[2]][1] = 1            # (if it hasn't been destroyed)
                    
                    pop.p[p[0]][1:3] = [0, best[0]]     # set household off-market, assign new dwelling
                    U = util + self.U_c[x]              # calculate household utility as:
                                                        # current utility 'self.U_c' + marginal utility from new dwelling 'util'
                    pop.U[p[0]] = U                     # set new household utility

                    P_r = U - self.s0b0[util_max, x]    # caclulate new dwelling price from price-free household utility
                    dwell.B[best[0],0] = P_r            # set new dwelling price to B0_new
                    dwell.r[best[0]][1] = 0             # set new dwelling off-market
                    self.rec_tran.append(best[0])

                    # remove cleared household & dwelling from household-dwelling utility array, on-market lists, utility lists
                    self.p_r_utils = np.delete(self.p_r_utils, util_max, axis=0)
                    self.p_r_utils = np.delete(self.p_r_utils, x, axis=1)
                    self.p_mark = np.delete(self.p_mark, x, axis=0)
                    self.r_mark = np.delete(self.r_mark, util_max, axis=0)
                    self.U_c = np.delete(self.U_c, x, axis=0)
                    pop.p[p[0]][3] += 1   # number of times a household transacts/relocates

                else:
                    # no positive marginal utility for any dwelling -- no transaction for household
                    pop.p[p[0]][1] = 0    # take household off-market

                    # remove uncleared household from household-dwelling utility array
                    self.p_r_utils = np.delete(self.p_r_utils, x, axis=1)
                    self.p_mark = np.delete(self.p_mark, x, axis=0)
            else:
                # clearance not successful -- no transaction for household
                pop.p[p[0]][1] = 0    # take household off-market

                # remove uncleared household from household-dwelling utility array
                self.p_r_utils = np.delete(self.p_r_utils, x, axis=1)
                self.p_mark = np.delete(self.p_mark, x, axis=0)

    def mse_dist(self, locs):
        locs = np.array(locs*10, dtype=int)
        centric = np.array(dwell.centric*10, dtype=int)
        dists = np.linalg.norm(locs.T - centric.T, axis=0)/10
        mse = np.mean(dists**2)
        return mse

class error:
    def check(household, dwelling):
        """ 
        check(household, dwelling) can be used for cross-validating household IDs with dwelling IDs to ensure
            model performed without errors & check for other discrepancies.
        
        *** Not currently in use ***

        param household: household information array; pop object
        param dwelling: dwelling information array; dwell object
        """

if __name__ == '__main__':

    start = time.time()
    cases = {'A':'City Formation',
            'A50':'City Formation -- larger system (50km x 50km)',
            'B':'Effect of U0',
            'C':'Effects of Spatial Localization of Favored Dwelling Characteristics',
            'D':'Segregation - monocentric',
            'D_poly':'Segregation - polycentric',
            'D_lrg':'Segregation - large system',
            'E_popinc':'Supply and Demand - population increase',
            'E_popdec':'Supply and Demand - population decrease',
            'E_dwellinc':'Supply and Demand - dwelling increase',
            'E_dwelldec':'Supply and Demand - dwelling decrease',
            '9':'Custom'}
    print('Example cases:')
    for key, val in cases.items():
        print(key +': '+ val)

    case = input('Enter test case (default A): ')
    if len(case) == 0:
        case = 'A'          # example case

    if case not in cases.keys():
        print('!! INPUT CASE NOT VALID, EXITING !!')
        exit()

    sim = simulate(case)    # initialise the model & parameters
    sim.create()            # create households and dwellings
    sim.evolve()            # begin simulation and evolution through time

    