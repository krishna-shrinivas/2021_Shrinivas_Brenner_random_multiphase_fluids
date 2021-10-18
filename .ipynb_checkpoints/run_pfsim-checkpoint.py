#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 11:22:25 2021

@author: krishna
"""

# if __name__ == '__main__':


#==============================================================================
# packages
#==============================================================================
import numpy as np
import os
import sys
import time
import argparse
import datetime
sys.path.append('..')
from utils import input_parse, gen_c0, write_params,save_movie, key_func
from FFT import FFTSolver


###################################################
#parameters are read from input files here
###################################################
def main_sim(args):
    
    # Parse the input files
    input_parameters = input_parse(args.i);

    # Parse the output files
    outfolder =  str(args.o) + '/'
    # os.makedirs(outfolder,exist_ok=True)

    # Check if a changing parameter file is submitted
    # if p exists, corresponding pN line number should be there
    if args.p is not None:
        params = input_parse(args.p,params_flag =True);
        if args.pN is not None:
            pN_number = int(args.pN);
        else:
            pN_number = 0;
        
        key = list(params.keys())[0]
        input_parameters[key] = params[key][pN_number];
        print("Parameter name and value are {},{}".format(key,params[key][pN_number]))
    else:
        print('No external parameter given - using default parameters')        

        
    #   Assign the seed to start the simulations
    if 'seed' in input_parameters.keys():
        seed = int(input_parameters['seed']);
    else:
        seed = np.random.randint(1e8);
    
    print("Random seed for run is {}".format(seed))
    np.random.seed(seed=seed)

    #   Define key system parameters
    NCom =  int(input_parameters['NCom']);
    dim =  int(input_parameters['dim']);
    N =  int(input_parameters['N']);
    start =  int(input_parameters['start']);
    end =  int(input_parameters['end']);

    
    # physical parameters
    if 'lmbda' in input_parameters.keys():
        lmbda = float(input_parameters['lmbda'])
    else:
        lmbda = 5.0e-5
        
    if 'kappa_mag' in input_parameters.keys():
        kappa_mag = float(input_parameters['kappa_mag'])
    else:
        kappa_mag = 10.0

    if 'dt' in input_parameters.keys():
        dt = float(input_parameters['dt'])
    else:
        dt  = 1.0e-6

    if 'beta' in input_parameters.keys():
        beta = float(input_parameters['beta'])
    else:        
        beta =float(NCom)/float(NCom+1);
        input_parameters['beta'] = beta;

    if 'noise_strength' in input_parameters.keys():
        noise_strength = float(input_parameters['noise_strength'])
    else:        
        noise_strength = 0.3*beta/NCom;

    if 'timestep_flag' in input_parameters.keys():
        timestep_flag = bool(input_parameters['timestep_flag'])
    else:        
        timestep_flag = False;        
        
    
    if 'run_flag' in input_parameters.keys():
        run_flag = bool(input_parameters['run_flag']);
    else:
        run_flag = False;
        

    kappa = np.identity(NCom)*kappa_mag;
   
    # chi (i,j) interaction matrix
    chi_mean = float(input_parameters['chi_mean'])
    chi_std = float(input_parameters['chi_std'])

    chi = chi_std*np.random.randn(NCom,NCom) + chi_mean;
    np.fill_diagonal(chi, 0);
    for i in np.arange(len(chi)-1):
        for j in np.arange(i+1,len(chi)):
            chi[i,j] = chi[j,i]
            
    input_parameters['chi'] = chi;

    print('chi \n',chi)
    
    # chi(i,s) is vector of component solvent interactions
    chi_s_mean = float(input_parameters['chi_s'])
    chi_s = chi_s_mean* np.ones((1,NCom))
    input_parameters['chi_s'] = chi_s;

    # chi_s += chi_mean;
    # chi_s[0,-1] = 4.5;
    chi_s = np.repeat(chi_s,NCom,0)
    
    # polymer lengths is vector of polymerizations
    r_mu = float(input_parameters['r_mu']);
    #     r_sigma = 1.0;
    #     r = r_sigma*np.random.randn(N,1) +r_mu ;
    #     r = r_sigma*np.random.lognormal(r_mu,r_sigma,size=(N,1)) ;
    
    r = r_mu*np.ones((NCom,1) )
    #     print(np.mean(r))
    r[r<1] = 1;
    input_parameters['r'] = r;

    print('Polymer lengths \n',r.T)
    
        
    kon = float(input_parameters['kon']);
    koff = kon*NCom/beta;
    
    g = np.zeros((NCom,NCom))
    np.fill_diagonal(g, beta/NCom)
    J = chi+ np.identity(NCom)*NCom/(beta*r) + 1/(1-beta)  - chi_s - chi_s.T
    input_parameters['J'] = J;
    print('Jacobian \n',J)
    wJ, vJ = np.linalg.eig(J)
    input_parameters['wJ'] = sorted(wJ);

    print('Eigen values \n',sorted(wJ))
    Jeff = np.matmul(g,J);
    wJeff,vJeff = np.linalg.eig(Jeff)
    print('Eigen values of modified J \n',sorted(wJeff))
    
    # output control
    if 'outPrint' in input_parameters.keys():
        outPrint = int(input_parameters['outPrint'])
    else:
        outPrint = 1e4

    if 'outSave' in input_parameters.keys():
        outSave = int(input_parameters['outSave'])
    else:        
        outSave  = 1e4
   
        

    # Check if time-step needs to be adaptively shifted
    if timestep_flag:
        dt = dt *(1.0/(r_mu)**1.5)*(0.5/abs(min(-0.5,min(wJeff))))
        print("Renormalized time-step is {}".format(dt))
        input_parameters['dt'] = dt;
        
    if run_flag:
        

        steps = end - start
        T   = steps * dt
        c_init = beta*np.ones((NCom))/NCom
        c0 = gen_c0(c_init, NCom, N)

#         print(c_init)

        outfile_nc = ["N","NCom","start","end","chi_mean","chi_std","lmbda","dt","kappa_mag","beta","kon","r_mu"]
        outfolder_sp = ""
        for i in outfile_nc:
            outfolder_sp += str(i) + "_" + str(input_parameters[i]) + "_"


        root =  outfolder + '/' + str(datetime.date.today()).replace('-','') + '/' + outfolder_sp +  '/' + str(seed) + '/'
        os.makedirs(root,exist_ok=True)
        print(root)
        write_params(root + '/all_params.txt',input_parameters)

        os.makedirs(root + "/Mesh/",exist_ok=True)
        np.save(root+ "/Mesh/"+'c-'+str(start)+'.npy', c0)

        with open(root+ "/stats.txt", 'w+') as stats:
            str_header = ["step","t","dt"]
            for i in range(NCom+1):
                str_header.append("c"+str(i)+"_max")
                str_header.append("c"+str(i)+"_min")
            str_header.append('t_sim')
            stats.write("\t".join(str_header) + "\n")


        c_all_temp = np.concatenate((c0,np.reshape(np.sum(c0,axis=0)*-1+ 1,(1,N,N))),axis=0)

        with open(root+ "/stats.txt", 'a') as stats:
            output_vars = [0,0,dt];
            for i in range(c_all_temp.shape[0]):
                output_vars.append(c_all_temp[i].max())
                output_vars.append(c_all_temp[i].min())
            output_vars.append('0')

            stats.write("\t".join([str(it) for it in output_vars]) + "\n")

        ###################################################
        #solving
        ###################################################

        # from FFT_nV_3D import FFTSolver

        Solver1 = FFTSolver(c_init=c0,
                            chiMat=chi, lmbda=lmbda,
                            dt=dt, T=T, N=N,
                            start=start, root=root,kappa=kappa,kon=kon,koff=koff,chis=chi_s,r=r)
        os.makedirs(root + "/Images/",exist_ok=True)

        Solver1.save_images(c_all_temp,start)
        cFFT3 = Solver1.solve(outPrint=outPrint, outSave=outSave)
        save_movie(root+"/Images/",NCom+1)
    ###################################################
    #end
    ###################################################




if __name__ == "__main__":
    """
        Function is called when python code is run on command line and calls main_sim
        to initialize the simulation
    """
    parser = argparse.ArgumentParser(description='Take output filename to run main_sim simulations')
    parser.add_argument('--i',help="path to input params file", required = True);
    parser.add_argument('--p',help="Name of parameter file", required = False);
    parser.add_argument('--pN',help="Parameter number from file (indexed from 1)", required = False);

    parser.add_argument('--o',help="Name of output folder", required = True);
    args = parser.parse_args();

    main_sim(args);