import os
import ndjson
import trajnettools
import numpy as np
import pysparkling
from joblib import Parallel, delayed
import trajnetbaselines
from interactions import Interaction
import random

import matplotlib.pyplot as plt

def get_type(scene):
    '''
    :param scene: All trajectories as TrackRows
    :return: The type of the traj
    '''

    ## Params
    static_threshold = 1.0
    linear_threshold = 0.2
    
    ## Interactions
    inter_pos_range = 15
    inter_dist_thresh = 5

    ## Group
    grp_dist_thresh = 2.0
    grp_std_thresh = 0.5

    ## Get xy-coordinates from trackRows
    scene_xy = trajnettools.Reader.paths_to_xy(scene)

    ## Type 1
    def euclidean_distance(row1, row2):
        """Euclidean distance squared between two rows."""
        return np.sqrt((row1.x - row2.x) ** 2 + (row1.y - row2.y) ** 2)

    ## Type 2 
    def linear_system(scene):
        '''
        return: True if the traj is linear according to Kalman
        '''
        kalman_prediction = trajnetbaselines.kalman.predict(scene)
        return trajnettools.metrics.average_l2(paths[0], kalman_prediction) 

    ## Type 3
    def interaction(rows, pos_range=15, dist_thresh=5):
        '''
        :return: Determine if someone is in the triangle
        '''
        return Interaction.check_interaction(rows, pos_range, dist_thresh)

    ## Type 4 = not(Type 1 or Type 2 or Type 3)

    ## Type 5
    def group(rows, dist_thresh=2, std_thresh=0.5):
        '''
        dist_thresh: Distance threshold to be withinin a group
        std_thresh: Std deviation threshold for variation of distance
        return: True if primary in a group
        '''     
        path = rows[:, 0]
        neigh_path = rows[:, 1:]
        dist_rel = np.linalg.norm((neigh_path[9:21] - path[9:21][:, np.newaxis, :]), axis=2)    
        mean_dist = np.mean(dist_rel, axis=0)
        std_dist = np.std(dist_rel, axis=0)

        group_matrix = (mean_dist < dist_thresh) & (std_dist < std_thresh)
        return np.any(group_matrix)

    ## Category Tags
    mult_tag = []
    # Static
    if euclidean_distance(ped_interest[0], ped_interest[-1]) < static_threshold:
        mult_tag.append(1)
    
    # Linear
    if linear_system(scene) < linear_threshold:
        mult_tag.append(2)

    # Interactions
    if interaction(scene_xy, inter_pos_range, inter_dist_thresh):
        mult_tag.append(3)

    # Non-Linear (No explainable reason)
    if mult_tag == []:
        mult_tag.append(4)

    # Group 
    if group(scene_xy, grp_dist_thresh, grp_std_thresh):
        mult_tag.append(5)        

    return mult_tag[0], mult_tag

def check_collision(scene):
        '''
        Skip the track if collision occurs between primanry and others
        return: True if collision occurs
        '''   
        ped_interest = scene[0]    
        for ped_other in scene[1:]:
            if trajnettools.metrics.collision(ped_interest, ped_other):
                return True
        return False

def write(rows, path, new_scenes, new_frames):
    pysp_tracks = rows.filter(lambda r: r.frame in new_frames).map(trajnettools.writers.trajnet)
    pysp_scenes = pysparkling.Context().parallelize(new_scenes).map(trajnettools.writers.trajnet)
    pysp_scenes.union(pysp_tracks).saveAsTextFile(path)

def trajectory_type(rows, path, fps, track_id=0):
    ## Read
    reader = trajnettools.Reader(path, scene_type='paths')
    scenes = [s for _, s in reader.scenes()]
    ## Filtered Frames and Scenes
    new_frames = set()
    new_scenes = []

    ###########################################################################
    # scenes_test helps to handle both test and test_private simultaneously
    # scenes_test correspond to Test
    ###########################################################################
    test = 'test' in path
    if test:
        path_test = path.replace('test_private', 'test')
        reader_test = trajnettools.Reader(path_test, scene_type='paths')
        scenes_test = [s for _, s in reader_test.scenes()]
        ## Filtered Test Frames and Test Scenes
        new_frames_test = set()
        new_scenes_test = []  

    ## Initialize Tag Stats to be collected
    tags = {1: [], 2: [], 3: [], 4: []}
    mult_tags = {1: [], 2: [], 3: [], 4: [], 5: []}
    tag_count = 0

    if not scenes:
        raise Exception('No scenes found')

    for index, scene in enumerate(scenes):
        ## Primary Path
        ped_interest = scene[0]

        ## Assert Test Scene length
        if test:
            assert len(scenes_test[index][0]) < 9, 'Scene Test not adequate length'

        ## Check Collision
        if check_collision(scene):
            continue

        ## Get Tag
        tag, mult_tag = get_type(scene)

        ## Update Tags
        tags[tag].append(tag_count)
        for tt in mult_tag:
            mult_tags[tt].append(tag_count)

        ## Filtered scenes and Frames
        new_frames |= set(ped_interest[i].frame for i in range(len(ped_interest)))
        # new_scenes.append(
        #     trajnettools.data.SceneRow(track_id, ped_interest[0].pedestrian,
        #                                ped_interest[0].frame, ped_interest[-1].frame, fps, tag))
        new_scenes.append(
            trajnettools.data.SceneRow(track_id, ped_interest[0].pedestrian,
                                       ped_interest[0].frame, ped_interest[-1].frame, fps, mult_tag))

        ## Append to list of scenes_test as well if Test Set
        if test:
            new_frames_test |= set(ped_interest[i].frame for i in range(len(scenes_test[index][0])))
            new_scenes_test.append(
                trajnettools.data.SceneRow(track_id, ped_interest[0].pedestrian,
                                           ped_interest[0].frame, ped_interest[-1].frame, fps, 0))

        track_id += 1


    # Writes the Final Scenes and Frames
    write(rows, path, new_scenes, new_frames)
    if test:
        write(rows, path_test, new_scenes_test, new_frames_test) 

    return track_id