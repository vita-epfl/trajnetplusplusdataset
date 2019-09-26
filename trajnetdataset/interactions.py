import argparse
import math
import numpy as np

import matplotlib.pyplot as plt
   
def compute_velocity_interaction(path, neigh_path, time_param=(9, 21, 9, 3)):
    ## Computes the angle between velocity of neighbours and velocity of pp

    T_OBS, T_SEQ, T_INT, T_STR = time_param 

    prim_vel = path[T_INT:T_SEQ] - path[T_INT-T_STR:T_SEQ-T_STR]
    theta1 = np.arctan2(prim_vel[:,1], prim_vel[:,0])
    neigh_vel = neigh_path[T_INT:T_SEQ] - neigh_path[T_INT-T_STR:T_SEQ-T_STR]
    vel_interaction = np.zeros(neigh_vel.shape[0:2])
    sign_interaction = np.zeros(neigh_vel.shape[0:2])

    for n in range(neigh_vel.shape[1]):
        theta2 = np.arctan2(neigh_vel[:, n, 1], neigh_vel[:, n, 0])
        theta_diff = (theta2 - theta1) * 180 / np.pi
        theta_diff = theta_diff % 360
        theta_sign = theta_diff > 180
        sign_interaction[:, n] = theta_sign
        vel_interaction[:, n] = theta_diff       
    return vel_interaction, sign_interaction


def compute_theta_interaction(path, neigh_path, time_param=(9, 21, 9, 3)):
    ## Computes the angle between line joining pp to neighbours and velocity of pp

    T_OBS, T_SEQ, T_INT, T_STR = time_param

    prim_vel = path[T_INT:T_SEQ] - path[T_INT-T_STR:T_SEQ-T_STR]
    theta1 = np.arctan2(prim_vel[:,1], prim_vel[:,0])
    rel_dist = neigh_path[T_INT:T_SEQ] - path[T_INT:T_SEQ][:, np.newaxis, :]
    theta_interaction = np.zeros(rel_dist.shape[0:2])
    sign_interaction = np.zeros(rel_dist.shape[0:2])

    for n in range(rel_dist.shape[1]):
        theta2 = np.arctan2(rel_dist[:, n, 1], rel_dist[:, n, 0])
        theta_diff = (theta2 - theta1) * 180 / np.pi
        theta_diff = theta_diff % 360
        theta_sign = theta_diff > 180
        sign_interaction[:, n] = theta_sign
        theta_interaction[:, n] = theta_diff
    return theta_interaction, sign_interaction

def compute_dist_rel(path, neigh_path, time_param=(9, 21, 9, 3)):
    ## Distance between pp and neighbour

    T_OBS, T_SEQ, T_INT, T_STR = time_param
    dist_rel = np.linalg.norm((neigh_path[T_INT:T_SEQ] - path[T_INT:T_SEQ][:, np.newaxis, :]), axis=2)
    return dist_rel


def compute_interaction(theta_rel_orig, dist_rel, angle, dist_thresh, angle_range):
    ## Interaction is defined as 
    ## 1. distance < threshold and 
    ## 2. angle between velocity of pp and line joining pp to neighbours

    theta_rel = np.copy(theta_rel_orig)
    angle_low = (angle - angle_range) 
    angle_high = (angle + angle_range) 
    if (angle - angle_range) < 0 :
        theta_rel[np.where(theta_rel > 180)] = theta_rel[np.where(theta_rel > 180)] - 360
    if (angle + angle_range) > 360 :
        raise ValueError
    interaction_matrix = (angle_low < theta_rel) & (theta_rel <= angle_high) & (dist_rel < dist_thresh) & (theta_rel < 500) == 1
    return interaction_matrix


def check_interaction(rows, pos_range=15, dist_thresh=5, choice='pos', pos_angle=0,  vel_angle=0, vel_range=15, output='all'):    

    path = rows[:, 0]
    neigh_path = rows[:, 1:]
    theta_interaction, sign_interaction = compute_theta_interaction(path, neigh_path)
    vel_interaction, sign_vel_interaction = compute_velocity_interaction(path, neigh_path)
    dist_rel = compute_dist_rel(path, neigh_path)
    
    ## str choice
    if choice == 'pos':
        interaction_matrix = compute_interaction(theta_interaction, dist_rel, pos_angle, dist_thresh, pos_range)

    elif choice == 'vel':
        interaction_matrix = compute_interaction(vel_interaction, dist_rel, vel_angle, dist_thresh, vel_range)

    elif choice == 'bothpos':
        interaction_matrix = compute_interaction(theta_interaction, dist_rel, pos_angle, dist_thresh, pos_range) \
                             & compute_interaction(vel_interaction, dist_rel, vel_angle, dist_thresh, vel_range)
        
    elif choice == 'bothvel':  
        interaction_matrix = compute_interaction(theta_interaction, dist_rel, pos_angle, dist_thresh, pos_range) \
                             & compute_interaction(vel_interaction, dist_rel, vel_angle, dist_thresh, vel_range)  
    else:
        raise NotImplementedError 

    if output == 'matrix':
        return interaction_matrix

    return np.any(interaction_matrix)

def interaction_length(interaction_matrix, length=1):
    interaction_sum = np.sum(interaction_matrix, axis=0)
    return interaction_sum >= length

def lf(rows, pos_range=15, dist_thresh=5):
    interaction_matrix = check_interaction(rows, pos_range=pos_range, dist_thresh=dist_thresh, choice='bothpos', output='matrix')
    interaction_index = interaction_length(interaction_matrix, length=5)
    return np.any(interaction_index)

def ca(rows, pos_range=15, dist_thresh=5):
    interaction_matrix = check_interaction(rows, pos_range=pos_range, dist_thresh=dist_thresh, choice='bothpos',  vel_angle=180, output='matrix')
    interaction_index = interaction_length(interaction_matrix, length=1)
    return np.any(interaction_index)

def group(rows, dist_thresh=0.8, std_thresh=0.2):
    interaction_index = check_group(rows, dist_thresh, std_thresh)        
    return np.any(interaction_index)

def get_interaction_type(rows, pos_range=15, dist_thresh=5):    
    interaction_type = []
    if lf(rows, pos_range, dist_thresh):
        interaction_type.append(1)
    if ca(rows, pos_range, dist_thresh):
        interaction_type.append(2)
    if group(rows):
        interaction_type.append(3)
    if interaction_type == []:
        interaction_type.append(4)    
    return interaction_type

def check_group(rows, dist_thresh=0.8, std_thresh=0.2):
    ## Identify Groups
    ## dist_thresh: Distance threshold to be withinin a group
    ## std_thresh: Std deviation threshold for variation of distance

    path = rows[:, 0]
    neigh_path = rows[:, 1:]

    ## Horizontal Position
    interaction_matrix_1 = check_interaction(rows, pos_angle=90, pos_range=45, output='matrix')
    interaction_matrix_2 = check_interaction(rows, pos_angle=270, pos_range=45, output='matrix')
    neighs_side = np.any(interaction_matrix_1, axis=0) | np.any(interaction_matrix_2, axis=0)

    ## Distance Maintain
    dist_rel = np.linalg.norm((neigh_path - path[:, np.newaxis, :]), axis=2)
    mean_dist = np.mean(dist_rel, axis=0)
    std_dist = np.std(dist_rel, axis=0)

    group_matrix = (mean_dist < dist_thresh) & (std_dist < std_thresh) & neighs_side

    return group_matrix
