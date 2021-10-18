#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 11:27:59 2021

@author: krishna
"""
import numpy as np
import moviepy.editor as mp
import os 
from pygifsicle import optimize as gif_optimize
import zipfile
import subprocess

def input_parse(filename,params_flag=False):
    """
    Parses input files (filename is path to input parameters or param_list file)
    params_flag toggles between input_params vs param_list
    """

    input_parameters  ={};
    with open(filename, 'r') as f:
        count = 0;

        for line in f:
            line=line.strip();
            if line:
                if line.find('#') == -1:
                    if not params_flag:
                        var_name,var_value = line.split(',');
                        if (var_value).lower() != 'none':
                            input_parameters[var_name] = float(var_value);
                    else:
                        if count==0:
                            var_name = line.strip('\n');
                            input_parameters[var_name] = [];
                            count+=1;
                        else:
                            input_parameters[var_name].append(float(line.strip('\n')))
    return input_parameters;


def gen_c0(c_init, M, N,noise_strength=0.01):
    c0_init = np.array(c_init)
    cmin = c0_init.min()
    noise_initial = noise_strength * cmin * np.random.uniform(-1,1,(M, N, N))
    if M>1:
        noise_initial -= noise_initial.mean(0)
    c0 = c0_init[:, np.newaxis, np.newaxis] + noise_initial
    return c0;

def write_params(file_output,input_params):
    """
    write_input_params writes all the input information from
    *input_params* into *file_output* in the same syntax
    """

    with open(file_output,'w+') as f:
        for key in input_params.keys():
            f.write( ''.join(key)+','+str(input_params[key])+'\n');
    f.close()

def key_func(x):
    """
    Input   =   List of PNG image file paths (x)
    Output  =   List of filenames without extensions (to sort)
    """
    return int(x.split('-')[-1].rstrip('.png'))

def save_movie(dir_name,NCom,movie_name ='movie',duration=0.1):
    """
    Function generates 2D movies (gif & MP4) from output images
    **Input parameters**
        -   dir_name    =   path to directory with image files
        -   movie_name  =   prefix for output movie file (default = movie)
        -   duration    =   time between 2 frames (default = 0.1)
    """
    all_files = os.listdir(dir_name);
    for i in range(NCom):
        relevant_files = [f for f in all_files if (f.find('c'+str(i)+'-')>-1 and f.endswith('.png'))]
        relevant_files = [dir_name + f for f in sorted(relevant_files,key=key_func)]    
        clip = mp.ImageSequenceClip(relevant_files,fps=1/duration);
        clip.write_gif(dir_name + 'c' + str(i) +'-' +  movie_name + '.gif',opt='nq');
        gif_optimize(dir_name + 'c' + str(i) +'-' +  movie_name + '.gif')

        with zipfile.ZipFile(dir_name+ 'c' + str(i) +'-' +'all_images.zip', 'w') as zipF:
            for file in relevant_files:
                zipF.write(file, compress_type=zipfile.ZIP_DEFLATED,arcname=os.path.basename(file))

    bash_cmd = 'rm '+ dir_name +'/*.png'
    res = subprocess.check_output(['bash','-c',bash_cmd])
        

# convert time scales
def convert_seconds_to_hms(seconds):
    hrs = int(seconds/3600.0)
    mins = int((seconds - hrs*3600)/60.)
    secs = seconds - hrs*3600 - mins*60.0
    return hrs, mins, secs