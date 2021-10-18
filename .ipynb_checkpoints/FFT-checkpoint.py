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

#==============================================================================
# FFT Solver Object for the time-stepping
#==============================================================================

class FFTSolver():
    
    def __init__(self, c_init, chiMat, lmbda, dt, T, N, start, root,kappa,kon,koff,chis,r):
        self.start = start
        self.root = root
        self.c0 = c_init.copy()
        self.NCom, _, _ = c_init.shape
        self.N = N
        self.gradMuX = c_init.copy()
        self.gradMuY = c_init.copy()
        self.gradcsX = c_init.copy()
        self.gradcsY = c_init.copy()

#         self.gradMuZ = c_init.copy()
        self.JXHat = np.zeros_like(c_init, dtype='complex')
        self.JYHat = np.zeros_like(c_init, dtype='complex')
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
#         self.kzj = self.kz*1j

    def fft(self, x, xHat):
        for i in range(self.NCom):
            xHat[i] = np.fft.fftn(x[i])


    def ifft(self, xHat, x):
        for i in range(self.NCom):
            x[i] = np.fft.ifftn(xHat[i]).real
            
        
    # calculate the chemical potential
    def cal_muHat(self, c, cHat):
        chiMat = self.chiMat
        k2 = self.k2
        lmbda = self.lmbda
        muHat = np.einsum('ij,jkl->ikl', chiMat, cHat*(1.0 - k2*lmbda))
        return muHat
    
    # calculate fluxes (multiplyied by Lij already)
    def cal_J(self, c, gradMu):
#         J0 = np.einsum('ijk,ijk->jk', c, gradMu)
        J  = ne.evaluate("c * (gradMu)")
        return J
    
    # calculate solvent fluxes (multiplyied by Lij already)
    def cal_sol_J(self, c, cs, gradMu):
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

        #         JZHat = self.JZHat
        
        # diffsion part
        NDiff = - ne.evaluate("k2 * cHat")/self.r[:,None]
        
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
        JsX = self.cal_sol_J(c,cs, gradcsX)
        JsY = self.cal_sol_J(c,cs, gradcsY)
#         JZ = self.cal_J(c, gradMuZ)
        # fluxes contribution
        self.fft(JsX, JsXHat)
        self.fft(JsY, JsYHat)
#         self.fft(JZ, JZHat)
        NsFlux = ne.evaluate("kxj*JsXHat + kyj*JsYHat")

        # calculate fluxes:
        JX = self.cal_J(c, gradMuX)
        JY = self.cal_J(c, gradMuY)
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
            cs = ax.contourf(c[i][:,:],cmap=plt.get_cmap(colors[i]),levels=levels)
            cbar = fig.colorbar(cs)
            # fig.savefig(fname = self.root + "/Images/"+ 'c' + str(i) + '-' +str(self.start+step)+'.pdf',format='pdf',dpi=300)
            fig.savefig(fname = self.root + "/Images/"+ 'c' + str(i) + '-' +str(self.start+step)+'.png',format='png',dpi=300)

            plt.close()
        
        print("Images saved at %d steps" % (step))
        
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
            c[c<1e-9] = 1e-9
            
            # This adds a simple first-order reaction process to the underlying reaction network
            
            c = c + (kon - koff*c)*dt/1.0;
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
                np.save(root + "/Mesh/" + 'c-'+str(start+step)+'.npy', c_all_temp)
                with open(root+ "/stats.txt", 'a') as stats:
                    output_vars = [step,tCur,dt];
                    for i in range(c_all_temp.shape[0]):
                        output_vars.append(c_all_temp[i].max())
                        output_vars.append(c_all_temp[i].min())
                    output_vars.append(time.time() - start_time)
                    
                    stats.write("\t".join([str(it) for it in output_vars]) + "\n")

        total_time = time.time() - start_time
        hrs, mins, secs = convert_seconds_to_hms(total_time)
        print("--- total time = %d hrs %d mins %.4f secs ---" % (hrs, mins, secs))

    

