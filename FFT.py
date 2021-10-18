#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 14:50:54 2021

@author: krishna
"""

import numpy as np
import numexpr as ne
import time 
import os
from utils import convert_seconds_to_hms
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
#==============================================================================
# FFT Solver Object for the time-stepping
#==============================================================================

class FFTSolver():
    
    def __init__(self, c_init, chiMat, lmbda, dt, T, N, start, root,kappa,kon,koff,chis,r,mobility_flag=0):
        self.start = start
        self.root = root
        self.c0 = c_init.copy()
        self.NCom, _, _ = c_init.shape
        self.N = N
        self.gradMuX = c_init.copy()
        self.gradMuY = c_init.copy()
        self.gradcsX = c_init.copy()
        self.gradcsY = c_init.copy()
        self.gradcX = c_init.copy()
        self.gradcY = c_init.copy()
        
#         self.gradMuZ = c_init.copy()
        self.JXHat = np.zeros_like(c_init, dtype='complex')
        self.JYHat = np.zeros_like(c_init, dtype='complex')
        self.JXDHat = np.zeros_like(c_init, dtype='complex')
        self.JYDHat = np.zeros_like(c_init, dtype='complex')
        self.JsXHat = np.zeros_like(c_init, dtype='complex')
        self.JsYHat = np.zeros_like(c_init, dtype='complex')
        
#         self.JZHat = np.zeros_like(c_init, dtype='complex')
        self.chiMat = chiMat
        self.r = r
        self.chis = chis
        self.lmbda  = lmbda
        self.kappa = kappa
        self.dt     = dt
        self.kon    = kon
        self.koff   = koff
        self.T      = T
        self.x  = np.linspace(0, 1, N+1)[:-1]
        self.xx, self.yy = np.meshgrid(self.x, self.x)
        self.dx = 1.0/N
        self.k  = 2 * np.pi * np.fft.fftfreq(N, self.dx)
        self.kx = self.k.reshape(-1,1)
        self.ky = self.k.reshape(1,-1)
#         self.kz = self.k.reshape(1,1,-1)
        self.k2 = self.kx**2 + self.ky**2
        self.k4 = self.kx**4 + self.ky**4
        self.kxj = self.kx*1j
        self.kyj = self.ky*1j
        
        self.mobility_flag = bool(mobility_flag)
#         self.kzj = self.kz*1j

    def fft(self, x, xHat):
        """
        Parameters
        ----------
        x : np.matrix of size (NCom,N,N)
            Contains the concentrations of
            all the species over the lattice
        xHat : np.matrix of size (NCom, N,N)
            FFT of the concentrations

        Returns
        -------
        None.

        """
        for i in range(self.NCom):
            xHat[i] = np.fft.fftn(x[i])


    def ifft(self, xHat, x):
        for i in range(self.NCom):
            x[i] = np.fft.ifftn(xHat[i]).real
            
        
    # calculate fluxes (multiplyied by Lij already)
    def cal_J(self, c, gradMu,mobility_flag=False):
        if mobility_flag:
            J  = ne.evaluate("c *(1-c)* (gradMu)")
        else:
            J  = ne.evaluate("c * (gradMu)")
        return J
    
    # calculate solvent fluxes (multiplyied by Lij already)
    def cal_sol_J(self, c, cs, gradMu,mobility_flag=False):

        if mobility_flag:
            J  = ne.evaluate("(c * (1-c) * (gradMu))/cs")
        else:
            J  = ne.evaluate("(c * (gradMu))/cs")
        return J
    

    def cal_NHat(self, c, cHat, A):
        k2 = self.k2
        k4 = self.k4
        kxj = self.kxj
        kyj = self.kyj
#         kzj = self.kzj
        
        #solvent concentrations
        cs = np.repeat(np.reshape(np.sum(c,axis=0)*-1+ 1,(1,self.N,self.N)) ,self.NCom,0);
        csHat = np.zeros_like(c, dtype='complex')
        self.fft(cs,csHat)
        
        gradMuX = self.gradMuX
        gradMuY = self.gradMuY
        
        gradcsX = self.gradcsX
        gradcsY = self.gradcsY    
        
#         gradMuZ = self.gradMuZ
        JXHat = self.JXHat
        JYHat = self.JYHat
        JsXHat = self.JsXHat
        JsYHat = self.JsYHat
        JXDHat = self.JXDHat
        JYDHat = self.JYDHat
        #         JZHat = self.JZHat
        
        # diffsion part
        
        if not self.mobility_flag:
            NDiff = - ne.evaluate("k2 * cHat")/self.r[:,None]
        else:
            
        # calculating diffusion coefficient the long-way
            gradcX = self.gradcX
            gradcY = self.gradcY
            self.ifft(ne.evaluate("kxj * cHat"), gradcX)
            self.ifft(ne.evaluate("kyj * cHat"), gradcY)    
            JDX = ne.evaluate('gradcX*(1-c)')
            JDY = ne.evaluate('gradcY*(1-c)')
            # fluxes contribution
            self.fft(JDX, JXDHat)
            self.fft(JDY, JYDHat)
    #         self.fft(JZ, JZHat)
            NDiff = ne.evaluate("kxj*JXDHat + kyj*JYDHat")       

        
        # implicit part
        NImp  = ne.evaluate("A * k4 * cHat")
        
        # fluxes part
        lmbda = self.lmbda
        chiMat = self.chiMat
        chis = self.chis
#         cHatGrad = ne.evaluate("cHat*(1.0 - k2*lmbda)")
        cHatGrad = ne.evaluate("cHat*1.0")
        kappa_term = ne.evaluate("cHat*k2*lmbda")

        muHat = (chiMat.dot(cHatGrad.reshape(self.NCom, -1))).reshape(self.NCom,self.N, self.N) 
        muHat = muHat + (self.kappa.dot(kappa_term.reshape(self.NCom, -1))).reshape(self.NCom,self.N, self.N)
        muHat = muHat - (chis.dot(cHatGrad.reshape(self.NCom, -1))).reshape(self.NCom, self.N, self.N)
                
        chis_d = np.diag(chis[:,0])
        muHat  = muHat + (chis_d.dot(csHat.reshape(self.NCom, -1))).reshape(self.NCom, self.N, self.N)
        #muHat   = np.einsum('ij,jklm->iklm', chiMat, cHatGrad)
        
        # gradients -> depend on spatial dim
        muKxHat = ne.evaluate("kxj * muHat")
        self.ifft(muKxHat, gradMuX)
        self.ifft(ne.evaluate("kyj * muHat"), gradMuY)
#         self.ifft(ne.evaluate("kzj * muHat"), gradMuZ)

        # gradients --> from solvent flux terms
        csKxHat = ne.evaluate("kxj*csHat")
        
        self.ifft(csKxHat, gradcsX)
        self.ifft(ne.evaluate("kyj*csHat"), gradcsY)
        
        # calculate solvent fluxes
        JsX = self.cal_sol_J(c,cs, gradcsX,self.mobility_flag)
        JsY = self.cal_sol_J(c,cs, gradcsY,self.mobility_flag)
#         JZ = self.cal_J(c, gradMuZ)
        # fluxes contribution
        self.fft(JsX, JsXHat)
        self.fft(JsY, JsYHat)
#         self.fft(JZ, JZHat)
        NsFlux = ne.evaluate("kxj*JsXHat + kyj*JsYHat")

        # calculate fluxes:
        JX = self.cal_J(c, gradMuX,self.mobility_flag)
        JY = self.cal_J(c, gradMuY,self.mobility_flag)
#         JZ = self.cal_J(c, gradMuZ)
        # fluxes contribution
        self.fft(JX, JXHat)
        self.fft(JY, JYHat)
#         self.fft(JZ, JZHat)
        NFlux = ne.evaluate("kxj*JXHat + kyj*JYHat")
        return ne.evaluate("NFlux + NDiff + NImp - NsFlux")

    def save_images(self,c,step):
        colors = ["Greens","Oranges","Blues","Reds","Greys","Purples","PuRd","BuGn","Greens","Oranges","Blues","Reds","Greys","Purples","PuRd","BuGn"]

        levels = np.linspace(0.0, 1.0, 15)
        for i in np.arange(c.shape[0]):
            fig,ax = plt.subplots()
            cs = ax.contourf(c[i][:,:],cmap=plt.get_cmap(colors[(i%10)]),levels=levels)
            cbar = fig.colorbar(cs)
            # fig.savefig(fname = self.root + "/Images/"+ 'c' + str(i) + '-' +str(self.start+step)+'.pdf',format='pdf',dpi=300)
            fig.savefig(fname = self.root + "/Images/"+ 'c' + str(i) + '-' +str(self.start+step)+'.png',format='png',dpi=300)

            plt.close()
        
        print("Images saved at %d steps" % (step))
        
    def save_inferred_phases(self,clabels,nphases,step):
        
        if nphases<10:
            cmapc = 'Set3'
        elif nphases<20:
            cmapc ='tab20'
        else:
            cmapc = 'inferno'
        levels = np.linspace(0.0, 1.0, nphases+1)
        fig,ax = plt.subplots()
        cs = ax.contourf(clabels.astype(float)/float(nphases-1),cmap=plt.get_cmap("ocean"),levels=levels)
        cbar = fig.colorbar(cs)
        fig.savefig(fname = self.root + "/Images/"+ 'c' + str(self.NCom+1) + '-' +str(self.start+step)+'.png',format='png',dpi=300)
        plt.close()
        
        
    def compute_gradient_filter(self,c):
        """
        

        Parameters
        ----------
        c : (Ncom*N*N)
            Concentrations of all species on lattice.
        filter : TYPE, optional
            DESCRIPTION. The default is 0.001.

        Returns
        -------
        gradient : (N*N) 
            magnitude of largest gradient
        
        """
        
        dcdx = np.zeros_like(c)
        dcdy = np.zeros_like(c)
        for N in range(c.shape[0]):
            for i in range(c.shape[1]):
                dcdx[N,i,:] =np.gradient(c[N,i,:])
                dcdy[N,:,i] = np.gradient(c[N,:,i])
        gradient = np.amax(dcdx**2 + dcdy**2,axis=0)

        return(gradient)



    def return_PCA_phases(self,c,flatten=True,centered=True,filter_thres=10.0,phase_thresh=9e-3):
        """
            Pass the observation matrix in the form
            (nsamples,nfeatures) with an optional flag of mean centering
            + std normalization (default is True).
            
            Function calculates pc_scores, eig_vals, and eig_vecs
            and uses this information to subsequently calculate the # of phases
        """
        
        gradient =  self.compute_gradient_filter(c)
        cre = np.reshape(c, (c.shape[0],-1))
        cre = cre[:,np.where(gradient.flatten()< gradient.max()/filter_thres + 1e-5)[0]].T
        
        if centered:
            mean_vec = np.mean(cre, axis=0)
            q_cent = (cre -mean_vec)/np.std(cre,axis=0)
        else:
            q_cent = cre
    
        #  If centering occurs, the calculated matrix is the correlation matrix
        
        cov_mat = (q_cent).T.dot(q_cent) / (q_cent.shape[0]-1)
    
        eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    
        key = np.argsort(eig_vals)[::-1]
        eig_vals, eig_vecs = eig_vals[key], eig_vecs[:, key]
        pc_scores =q_cent.dot(eig_vecs)
        
        nphases = sum(eig_vals>phase_thresh)
        kmeans = KMeans(n_clusters=nphases, random_state=0).fit(cre);
        return (eig_vals,eig_vecs,pc_scores,nphases,kmeans)



    def solve(self, outPrint, outSave):
        start = self.start
        root = self.root

        kon = self.kon
        koff = self.koff
        c = self.c0.copy()
        cHat = np.zeros_like(c, dtype='complex')
        self.fft(c, cHat)
        dt = self.dt
        T  = self.T
        tCur = 0.0
        A = 1.0 * self.chiMat.max() * self.lmbda
        #A = 0.0
        k4 = self.k4
        start_time = time.time()
        step = 0
        Ainv = 1.0/(1 + A * k4 * dt)
        print("--- Using %d of threads to calculate numpy fft---" % (ne.nthreads))
        while (tCur < T + dt/2.):
            tCur += dt
            step += 1
            cnHat = cHat
            cn = c
            ncHat = self.cal_NHat(cn, cnHat, A)
            cHat = ne.evaluate( "(cnHat + dt * ncHat) * Ainv" )
            self.ifft(cHat, c)
            
            # Forces concentrations less than 1e-9 to become 1e-9 --> causes error in simulation
            c[c<1e-9] = 1e-9;
            
            # This adds a simple first-order reaction process to the underlying reaction network
            
            c = c + (kon - koff*c)*dt/1.0;
            
            # Need to write a function that basically identifies positions sum(c) is large and then normalize it
            # c[:,np.sum(c,axis=0)>(1e-1e-9*self.NCom)] /=  
            self.fft(c,cHat)

            if np.isnan(c.max()):
                print("Simulation ended because of NAN values")
                np.save(root + 'c-'+str(start + step)+'.npy', cn)
                break
            
            if (step%outPrint == 0):
                dtime = time.time() - start_time
                print("%d steps finished with max c %.4f, using %.4f seconds/step" % (step, c.max(), dtime/step))
#                 if (start+step) < outSave:
#                     np.save(root + 'c-'+str(start+step)+'.npy', c)
#                     save_images(c,start,step,root)

            if (step%outSave == 0):
                print("----------------------step %d saved---------- ------------------" % (step+start))
                c_all_temp = np.concatenate((c,np.reshape(np.sum(c,axis=0)*-1+ 1,(1,self.N,self.N))),axis=0)
                self.save_images(c_all_temp,step)
                (eig_vals,eig_vecs,pc_scores,nphases,kmeans) = self.return_PCA_phases(c_all_temp,centered=False);
                clabels_p = kmeans.predict(c_all_temp.reshape(c_all_temp.shape[0],-1).T).reshape(c_all_temp.shape[1],c_all_temp.shape[2])
                cl = np.zeros_like(clabels_p)
                count = 0;
                for i in np.argsort(kmeans.cluster_centers_[:,0]):
                    cl[np.where(clabels_p==i)] = count;
                    count+=1;
                
                self.save_inferred_phases(cl,nphases,step)
                np.save(root + "/Mesh/" + 'c-'+str(start+step)+'.npy', c_all_temp)
                np.save(root + "/Mesh/" + 'clusters-'+str(start+step)+'.npy', kmeans.cluster_centers_)

                with open(root+ "/stats.txt", 'a') as stats:
                    output_vars = [step,tCur,dt];
                    for i in range(c_all_temp.shape[0]):
                        output_vars.append(c_all_temp[i].max())
                        output_vars.append(c_all_temp[i].min())
                    output_vars.append(nphases)
                    output_vars.append(time.time() - start_time)

                    stats.write("\t".join([str(it) for it in output_vars]) + "\n")

        total_time = time.time() - start_time
        hrs, mins, secs = convert_seconds_to_hms(total_time)
        print("--- total time = %d hrs %d mins %.4f secs ---" % (hrs, mins, secs))

    

