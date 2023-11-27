# spatiotemporal-urban-model
This repository contains code to reproduce all figures from our paper:

Robinson PA, McInnes A, Sarkar S (2023) Spatiotemporal evolution of urban populations and housing: A dynamic utility-driven market-mediated model. PLOS ONE 18(4): e0282583. https://doi.org/10.1371/journal.pone.0282583


Abstract: A model of the spatiotemporal evolution of urban areas is developed that includes effects on house-hold utility of geography, population density, household preference for characteristics of dwellings and neighbors, and disposable income. Housing market evolution then results via transactions driven by increases in utility and changes in numbers of households and dwellings. Examples are presented of how the model can simulate formation of monocentric and polycentric urban areas, stratification by wealth, segregation due to preferences for housing or neighbors, and the balance of supply and demand. These results reproduce, unify, and extend the results of prior models that covered specialized aspects of these phenomena. Potential generalizations are discussed and further applications are suggested.



## Info 

/main/ contains code for the main model & associated visualisation package. The main model contains all necessary code to reproduce the example cases below. The visualisation function contains all code to produce relevant results + additional figures.

/example-cases/ contains code for all example cases within the paper (+ few variations not included within the paper). These are variations of the main model.




## Required directory structure to run the model

/market_dynamics_main.py - main model

/market_dynamics_visualisations.py - visualisation package

/data/output/ contains all data and relevant figures output from simulation. A directory with corresponding date and timestamp of simulation will be created to store the ouputs of each run of the model.

/data/input/ - allows for input of results into market_dynamics_visualisations.py from previous iterations of the model for further analysis of results & alternative plots.




## Instructions

this file should serve as instructions for use of the model.

`market_dynamics_main.py` contains the simulation component of the model, while `market_dynamics_visualisations.py` contains all plotting related functions to visualise the results. Case input is required at the end of the code (approx. line 1852) in market_dynamics.py, with the parameters of the example cases below pre-loaded. Case '9' allowing for custom specification of parameters and for extensions of the model. Most of the parameters for the model are contained within the 'simulate.model_parameters()' function (approx. line 1129). 

During model initialisation, households, dwellings and other relevant model arrays are randomly generated. Following from this, any relevent figures depicting the model specification are produced and saved. Iteration from t=0 to t=y is carried out where y is the number of years with typically 4 rounds transactions occurring per year. During each round of transactions, all household utilities and dwelling prices are calculated and updated prior to transaction, and the utilities and prices of households/dwellings that transact are also updated following transaction. The household and dwelling profile arrays are saved at each timestep. Following iteration of the model, all figures relevant to the specified case are produced and saved.

Code has thorough comments.




## Example cases

Case A (seed 2) - City Formation: effect of q0.q0. This demonstrates the effect of clustering via the characteristic for clustering q0, with single-multiple large cluster(s) forming in the default version. This case produces figure 7 within the paper. 

Case B (seed 69) - Effect of U0: effect of U0(r) term. This demonstrates the effect of clustering via the characteristic for clustering q0, with the U0(r) vector associated with dwellings providing a spatial preference for clustering. The default version has U0(r) specified at maximum at (12.5, 12.5) in a system with size R=25km and declining as a gaussian. A secondary effect visible in this case is the stratification of households via wealth (price sensitivity) s0, with the wealthiest dominating the centre of the cluster where U0(r) is at maximum. This case produces figures 8, 9, 10 within the paper.

Case C (seed 99) - Effects of Spatial Localization of Favored Dwelling Characteristics: effect of s.B (beyond s0.B0). This case produces figure 11 within the paper.

Case D (seed 70) - Segregation: effect of q.q (beyond q0.q0). This demonstrates the effect of segregation by characteristic type q1 which is sampled from [-1,+1] by default. Figures 12a-f show results for variations in q1 and neighbourhood size h. The default version has q1 = +-1 and h = 1km. This case produces figures 12,13, and the movie corresponding to figure 13 within the paper.

Case E (seed 69) - Supply and Demand (effect of non-zero epsilon, rho, beta, lambda; altering households N & dwellings M respectively). This case contains 4 modes for population increase/decrease and dwelling increase/decrease. This demonstrates the effects of supply and demand on the density of households/dwellings, the price of dwellings, the utility of households. This case produces figure 14 within the paper.
