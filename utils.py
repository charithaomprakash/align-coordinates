#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 17:08:43 2023

@author: compraka
"""

import numpy as np
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import tqdm

#Using the Kabsch-Umeyama algorithm to align and compare the similarity between two sets of points.
#It finds the optimal translation, rotation and scaling by minimizing te root mean sqare deviation 
#of the point pair

def align_2d_ku_algorithm(ref_keypoints, keypoints):
    
    assert ref_keypoints.shape == keypoints.shape
    n, m = ref_keypoints.shape
    
    ref_kpts_mean = np.mean(ref_keypoints, axis=0)
    kpts_mean = np.mean(keypoints, axis=0)
    
    ref_kpts_cent = ref_keypoints - ref_kpts_mean
    kpts_cent = keypoints - kpts_mean
    
    ref_kpts_var = np.mean(np.linalg.norm(ref_kpts_cent, axis=1)**2)
    
    H = (ref_kpts_cent.T @ kpts_cent) / n
    U, D, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U) * np.linalg.det(Vt))
    S = np.diag([1] * (m - 1) + [d])
    
    rot = U @ S @ Vt
    scale = ref_kpts_var / np.trace(np.diag(D) @ S)
    translate = ref_kpts_mean - scale * rot @ kpts_mean
    
    aligned_kpts = np.array([translate + scale * rot @ b for b in keypoints])
    
    #Plot the reference keypoints, keypoints and aligned keypoints
    plt.figure()
    plt.plot(ref_keypoints[:, 0], ref_keypoints[:, 1], 'r', label='Reference keypoints')
    plt.plot(keypoints[:, 0], keypoints[:, 1], 'k', label='Keypoints to align')
    plt.plot(aligned_kpts[:, 0], aligned_kpts[:, 1], '--', label='Aligned keypoints')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.legend()
    
    return rot, scale, translate
    

def nan_helper(y):
    return np.isnan(y), lambda z:z.nonzero()[0]


def interpol(arr):
    y = np.transpose(arr)
    nans, x = nan_helper(y[0])
    y[0][nans] = np.interp(x(nans), x(~nans), y[0][~nans])
    nans, x = nan_helper(y[1])
    y[1][nans] = np.interp(x(nans), x(~nans), y[1][~nans])
    arr = np.transpose(y)
    
    return arr


def data_filter(x_coord, y_coord, likelihood, ref_keypoints, keypoints,
                rot, scale, translate, col, align=True):
    
    data_mat = np.column_stack((x_coord, y_coord))
    accu_thrshold = 0.60
    
    nan_count=0
    for idx in range(data_mat.shape[0]):
        if likelihood[idx] < accu_thrshold:
            data_mat[idx, 0] = np.nan
            data_mat[idx, 1] = np.nan
            nan_count = nan_count + 1
        
    data_mat_ = interpol(data_mat)
    
    if align == True:
        mapped_xy = np.array([translate + scale * rot @ b for b in data_mat_])
        
        plt.figure()
        plt.plot(ref_keypoints[:, 0], ref_keypoints[:, 1], 'r', label='Reference keypoints')
        plt.plot(keypoints[:, 0], keypoints[:, 1], 'k', label='Keypoints to align')
        plt.scatter(mapped_xy[:, 0], mapped_xy[:, 1], c='r', label='Raligned body keypoint positions')
        plt.title(str(col))
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.legend()
        
    else:
        mapped_xy = data_mat_
        
    return mapped_xy


def align_trajectores(trajectories_folder_path, ref_keypoints_path, keypoints_path, 		              
                      save_csvs_folder, align=True,  file_type='*helper.csv*'):
    
    csv_files = sorted(glob.glob(os.path.join(trajectories_folder_path, file_type)))
    
    save_results_fold =  save_csvs_folder + '/aligned_outputs'
    
    if not os.path.exists(save_results_fold):
        os.mkdir(save_results_fold)
        print('Folder %s created!' % save_results_fold)
    else:
        print('Folder %s already exists!' % save_results_fold)
    
    ref_keypoints = np.load(ref_keypoints_path).T
    keypoints = np.load(keypoints_path).T
    rot, scale, translate = align_2d_ku_algorithm(ref_keypoints, keypoints)
    
    for csv_file in csv_files:
        filepath, name = os.path.split(csv_file)
        save_path = save_results_fold + '/' + name[:-4] + '_aligned.csv' 
        dataframe = pd.read_csv(csv_file, header=[0, 1], index_col=0)
        aligned_bds = []
        for col in np.unique(dataframe.columns.get_level_values(level=0)):
            temp = dataframe[col].loc[:, dataframe[col].columns.isin({'x', 'y', 'likelihood'})]
            
            data_mat_aligned = data_filter(temp['x'].values, temp['y'].values, temp['likelihood'].values, 
                                           ref_keypoints, keypoints, rot, scale, translate, 
                                           col, align=align)
            
            aligned_bds.append(pd.DataFrame(data_mat_aligned, columns=[col + '_x', col + '_y']))
            
        aligned_dataframe = pd.concat(aligned_bds, axis=1, ignore_index=False)
        aligned_dataframe.to_csv(save_path)
        
        
if __name__=='__main__':
    
    save_csvs_folder = r'/home/compraka/Documents/Projects/prosocial_behavior/results_all/aligned_DLC_csv/nostressor_2021_03_11'
    ref_keypoints_path = r'/home/compraka/Documents/Projects/prosocial_behavior/ROI_marking/reference_keypoints_victim.npy'
    keypoints_path = r'/home/compraka/Documents/Projects/prosocial_behavior/ROI_marking/nostressor_2021_03_11_keypoints_victim.npy'
    zone_coords_path = r'/home/compraka/Documents/Projects/prosocial_behavior/ROI_marking/rescue_zone_coordinates.npy'
    
    trajectories_folder_path = r'/home/compraka/Documents/Projects/prosocial_behavior/results_all/DLC/nostressor_2021_03_11'
    
    
    align_trajectores(trajectories_folder_path, ref_keypoints_path, keypoints_path, 		              
                      zone_coords_path, save_csvs_folder, align=True,  file_type='*helper.csv*')
