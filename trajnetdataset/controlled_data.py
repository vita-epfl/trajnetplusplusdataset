""" Generating Controlled data for pretraining collision avoidance """

import random
import argparse
import os
import itertools

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

import rvo2
import pickle
import socialforce
from socialforce.potentials import PedPedPotential
from socialforce.fieldofview import FieldOfView

def generate_circle_crossing(num_ped, sim=None, radius=4, mode=None): 
    positions = []
    goals = []
    speed = []
    agent_list = []
    if mode == 'trajnet':
        radius = 10 ## 10 (TrajNet++)
    for _ in range(num_ped):
        while True:
            angle = random.uniform(0, 1) * np.pi * 2
            # add some noise to simulate all the possible cases robot could meet with human
            px_noise = (random.uniform(0, 1) - 0.5)  ## human.v_pref
            py_noise = (random.uniform(0, 1) - 0.5)  ## human.v_pref
            px = radius * np.cos(angle) + px_noise
            py = radius * np.sin(angle) + py_noise
            collide = False
            for agent in agent_list:
                min_dist = 0.8
                if mode == 'trajnet':
                    min_dist = 2    ## min_dist ~ 2*human.radius + discomfort_dist ## 2 (TrajNet++)
                if norm((px - agent[0], py - agent[1])) < min_dist or \
                        norm((px - agent[2], py - agent[3])) < min_dist:
                    collide = True
                    break
            if not collide:
                break

        positions.append((px, py))
        goals.append((-px, -py))
        if sim is not None:
            sim.addAgent((px, py))
        velocity = np.array([-2 * px, -2 * py])
        magnitude = np.linalg.norm(velocity)
        init_vel = 1 * velocity / magnitude if magnitude > 1 else velocity
        speed.append([init_vel[0], init_vel[1]])
        agent_list.append([px, py, -px, -py])
    trajectories = [[positions[i]] for i in range(num_ped)]
    return trajectories, positions, goals, speed

def generate_orca_trajectory(sim_scene, num_ped, min_dist=3, react_time=1.5, end_range=1.0, mode=None):
    """ Simulating Scenario using ORCA """
    ## Default: (1 / 60., 1.5, 5, 1.5, 2, 0.4, 2)
    sampling_rate = 1

    ## Circle Crossing
    if sim_scene == 'circle_crossing':
        fps = 100
        sampling_rate = fps / 2.5
        sim = rvo2.PyRVOSimulator(1/fps, 10, 10, 5, 5, 0.3, 1)
        if mode == 'trajnet':
            sim = rvo2.PyRVOSimulator(1/fps, 4, 10, 4, 5, 0.6, 1.5) ## (TrajNet++)
        trajectories, _, goals, speed = generate_circle_crossing(num_ped, sim, mode=mode)
    else:
        raise NotImplementedError

    # run
    done = False
    reaching_goal_by_ped = [False] * num_ped
    count = 0
    valid = True
    ##Simulate a scene
    while not done and count < 6000:
        count += 1
        sim.doStep()
        reaching_goal = []
        for i in range(num_ped):
            if count == 1:
                trajectories[i].pop(0)
            position = sim.getAgentPosition(i)

            ## Append only if Goal not reached
            if not reaching_goal_by_ped[i]:
                if count % sampling_rate == 0:
                    trajectories[i].append(position)

            # check if this agent reaches the goal
            if np.linalg.norm(np.array(position) - np.array(goals[i])) < end_range:
                reaching_goal.append(True)
                sim.setAgentPrefVelocity(i, (0, 0))
                reaching_goal_by_ped[i] = True
            else:
                reaching_goal.append(False)
                velocity = np.array((goals[i][0] - position[0], goals[i][1] - position[1]))
                speed = np.linalg.norm(velocity)
                pref_vel = 1 * velocity / speed if speed > 1 else velocity
                sim.setAgentPrefVelocity(i, tuple(pref_vel.tolist()))
        done = all(reaching_goal)

    if not done or not are_smoothes(trajectories):
        valid = False

    return trajectories, valid, goals

def generate_sf_trajectory(sim_scene, num_ped, sf_params=[0.5, 2.1, 0.3], end_range=0.2):
    """ Simulating Scenario using SF """
    ## Default: (0.5, 2.1, 0.3)
    sampling_rate = 1

    ## Circle Crossing
    if sim_scene == 'circle_crossing':
        fps = 10
        sampling_rate = fps / 2.5
        trajectories, positions, goals, speed = generate_circle_crossing(num_ped)
    else:
        raise NotImplementedError

    initial_state = np.array([[positions[i][0], positions[i][1], speed[i][0], speed[i][1],
                               goals[i][0], goals[i][1]] for i in range(num_ped)])

    ped_ped = PedPedPotential(1./fps, v0=sf_params[1], sigma=sf_params[2])
    field_of_view = FieldOfView()
    s = socialforce.Simulator(initial_state, ped_ped=ped_ped, field_of_view=field_of_view,
                              delta_t=1./fps, tau=sf_params[0])

    # run
    reaching_goal = [False] * num_ped
    done = False
    count = 0

    #Simulate a scene
    while not done and count < 500:
        count += 1
        position = np.stack(s.step().state.copy())
        for i in range(len(initial_state)):
            if count % sampling_rate == 0:
                trajectories[i].append((position[i, 0], position[i, 1]))
            # check if this agent reaches the goal
            if np.linalg.norm(position[i, :2] - np.array(goals[i])) < end_range:
                reaching_goal[i] = True
            else:
                s.state[i, :4] = position[i, :4]
        done = all(reaching_goal)

    return trajectories, count


def getAngle(a, b, c):
    """
    Return angle formed by 3 points
    """
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return angle

def are_smoothes(trajectories):
    """
    Check if there is no sharp turns in the trajectories
    """
    is_smooth = True
    for i, _ in enumerate(trajectories):
        trajectory = np.array(trajectories[i])
        for j in range(0, len(trajectory[:, 0]) - 3):
            p1 = np.array([trajectory[j, 0], trajectory[j, 1]])
            p2 = np.array([trajectory[j+1, 0], trajectory[j+1, 1]])
            p3 = np.array([trajectory[j+2, 0], trajectory[j+2, 1]])

            angle = getAngle(p1, p2, p3)
            if angle <= np.pi / 2:
                is_smooth = False
                # plt.scatter(p1[0], p1[1], color='red', marker='X')
    return is_smooth

def find_collisions(trajectories, max_steps):
    """
    Look for collisions in the trajectories
    """
    for timestep in range(max_steps):
        positions = []
        for ped, _ in enumerate(trajectories):
            traj = np.array(trajectories[ped])
            if timestep < len(traj):
                positions.append(traj[timestep])

        # Check if distance between 2 points is smaller than 0.1m
        # If yes -> collision detected
        for combi in itertools.combinations(positions, 2):
            distance = (np.linalg.norm(combi[0]-combi[1]))
            if distance < 0.2:
                return True

    return False

def write_to_txt(trajectories, path, count, frame, dict_dest=None, goals=None):
    """ Write Trajectories to the text file """

    last_frame = 0
    with open(path, 'a') as fo:
        track_data = []
        for i, _ in enumerate(trajectories):
            for t, _ in enumerate(trajectories[i]):

                track_data.append('{}, {}, {}, {}'.format(t+frame, count+i,
                                                          trajectories[i][t][0],
                                                          trajectories[i][t][1]))

                if t == len(trajectories[i])-1 and t+frame > last_frame:
                    last_frame = t+frame
            if goals:
                dict_dest[count+i] = goals[i]

        for track in track_data:
            fo.write(track)
            fo.write('\n')

    return last_frame

def viz(trajectories, mode=None):
    """ Visualize Trajectories """
    for i, _ in enumerate(trajectories):
        trajectory = np.array(trajectories[i])
        plt.plot(trajectory[:, 0], trajectory[:, 1])

    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    if mode == 'trajnet':
        plt.xlim(-15, 15) ## TrajNet++
        plt.ylim(-15, 15) ## TrajNet++
    plt.show()
    plt.close()

def predict_all(input_paths, goals, n_predict=12):

    pred_length = n_predict

    fps = 100
    sampling_rate = fps / 2.5

    sim = rvo2.PyRVOSimulator(1/fps, 4, 10, 4, 5, 0.6, 1.5) ## (TrajNet++)
    trajectories = [[input_paths[i][-1]] for i, _ in enumerate(input_paths)]
    [sim.addAgent((p[-1][0],p[-1][1])) for p in input_paths]

    num_ped = len(trajectories)
    reaching_goal_by_ped = [False] * num_ped
    count = 0
    end_range = 1.0
    done = False

    for i in range(num_ped):
        velocity = np.array((input_paths[i][-1][0] - input_paths[i][-3][0], input_paths[i][-1][1] - input_paths[i][-3][1]))
        velocity = velocity/0.8
        sim.setAgentVelocity(i, tuple(velocity.tolist()))

        velocity = np.array((goals[i][0] - input_paths[i][-1][0], goals[i][1] - input_paths[i][-1][1]))
        speed = np.linalg.norm(velocity)
        pref_vel = 1 * velocity / speed if speed > 1 else velocity
        sim.setAgentPrefVelocity(i, tuple(pref_vel.tolist()))

    ##Simulate a scene
    while (not done) and count < sampling_rate * pred_length + 1:
        # print("Count: ", count)
        count += 1
        sim.doStep()
        reaching_goal = []
        for i in range(num_ped):
            if count == 1:
                trajectories[i].pop(0)
            position = sim.getAgentPosition(i)

            ## Append only if Goal not reached
            if not reaching_goal_by_ped[i]:
                if count % sampling_rate == 0:
                    trajectories[i].append(position)

            # check if this agent reaches the goal
            if np.linalg.norm(np.array(position) - np.array(goals[i])) < end_range:
                reaching_goal.append(True)
                sim.setAgentPrefVelocity(i, (0, 0))
                reaching_goal_by_ped[i] = True
            else:
                reaching_goal.append(False)
                velocity = np.array((goals[i][0] - position[0], goals[i][1] - position[1]))
                speed = np.linalg.norm(velocity)
                pref_vel = 1 * velocity / speed if speed > 1 else velocity
                sim.setAgentPrefVelocity(i, tuple(pref_vel.tolist()))

        done = all(reaching_goal)

    return trajectories

def evaluate_sensitivity(trajectories, goals, mode=None, ade_thresh=0.11, fde_thresh=0.2, iters=20):
    observation = np.array([trajectory[10:15] for trajectory in trajectories])
    observation = np.round(observation, 2)
    goals = np.array(goals)

    trajectories_re_list = []
    for k in range(iters):
        observation_re = add_noise(observation.copy())
        trajectories_re = predict_all(observation_re, goals)
        for m, _ in enumerate(trajectories_re):
            diff_ade =  np.mean(np.linalg.norm(np.array(trajectories[m][15:27]) - np.array(trajectories_re[m]), axis=1))
            diff_fde =  np.linalg.norm(np.array(trajectories[m][26]) - np.array(trajectories_re[m][-1]))
            if diff_ade > ade_thresh or diff_fde > fde_thresh:
                print("INVALID", diff_ade, diff_fde)
        trajectories_re_list.append(np.array(trajectories_re))

    visualize_sensitivity(trajectories, trajectories_re_list, mode=mode)

def visualize_sensitivity(trajectories, trajectories_pred_scenes, mode=None):
    """ Visualize Trajectories """
    plt.grid(linestyle='dotted')
    for i, _ in enumerate(trajectories):
        trajectory = np.array(trajectories[i])
        if i == 0:
            plt.plot(trajectory[:, 0], trajectory[:, 1], linestyle='solid',
                     color='black', marker='o', markersize=1.0, zorder=1.9)
        else:
            plt.plot(trajectory[:, 0], trajectory[:, 1], linestyle='None',
                     color='black', marker='o', markersize=1.0, zorder=0.9)

    for i, _ in enumerate(trajectories_pred_scenes):
        trajectory_set = np.array(trajectories_pred_scenes[i])
        for j, _ in enumerate(trajectory_set):
            trajectory = trajectory_set[j]
            plt.plot(trajectory[:, 0], trajectory[:, 1], linestyle='solid',
                     color='blue', alpha=0.4, linewidth=2)

    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    if mode == 'trajnet':
        plt.xlim(-7, 7) ## TrajNet++
        plt.ylim(-7, 7) ## TrajNet++
    plt.show()
    plt.close()

def add_noise(observation):
    ## Last Position Noise
    # observation[0][-1] += np.random.uniform(0, 0.04, (2,))

    ## Last Position Noise
    thresh = 0.005
    observation += np.random.uniform(-thresh, thresh, observation.shape)
    return observation

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--simulator', default='orca',
                        choices=('orca', 'social_force'))
    parser.add_argument('--simulation_scene', default='circle_crossing',
                        choices=('circle_crossing'))
    parser.add_argument('--style', required=False, default=None)
    parser.add_argument('--num_ped', type=int, default=10,
                        help='Number of ped in scene')
    parser.add_argument('--num_scenes', type=int, default=100,
                        help='Number of scenes')
    parser.add_argument('--test', default=False)
    parser.add_argument('--mode', default=None,
                        help='Keep trajnet for trajnet dataset generation')

    args = parser.parse_args()

    ##Decide the number of scenes & agents per scene
    num_scenes = args.num_scenes
    num_ped = args.num_ped
    mode = args.mode

    if not os.path.isdir('./data'):
        os.makedirs('./data')

    ## Text File To Write the Scene
    output_file = 'data/raw/controlled/'
    if args.test:
        output_file = output_file + 'test_'
    output_file = output_file \
                  + args.simulator + '_' \
                  + args.simulation_scene + '_' \
                  + str(num_ped) + 'ped_' \
                  + str(num_scenes) + 'scenes_' \
                  + args.style + '.txt'
    print(output_file)

    count = 0
    last_frame = -5

    dict_dest = {}

    for i in range(num_scenes):
        if mode == 'trajnet':
            num_ped = random.choice([4, 5, 6]) ## TrajNet++
        ## Print every 10th scene
        if (i+1) % 10 == 0:
            print(i)

        ##Generate scenes
        if args.simulator == 'orca':
            trajectories, valid, goals = generate_orca_trajectory(sim_scene=args.simulation_scene,
                                                                  num_ped=num_ped,
                                                                  min_dist=min_dist,
                                                                  react_time=react_time,
                                                                  mode=mode)
            ## To evaluate sensitivity of ORCA
            # evaluate_sensitivity(trajectories, goals, mode)

        elif args.simulator == 'social_force':
            trajectories, valid = generate_sf_trajectory(sim_scene=args.simulation_scene,
                                                         num_ped=num_ped,
                                                         sf_params=[0.5, 1.0, 0.1])
        else:
            raise NotImplementedError

        ## Visualizing scenes
        # viz(trajectories, mode=mode)

        ## Write if the scene is valid
        if valid:
            last_frame = write_to_txt(trajectories, output_file,
                                      count=count, frame=last_frame+5,
                                      dict_dest=dict_dest,
                                      goals=goals)
        count += num_ped

    ## Write Goal Dict of ORCA
    with open('dest_new/' + args.simulator + '_' \
                  + args.simulation_scene + '_' \
                  + str(num_ped) + 'ped_' \
                  + str(num_scenes) + 'scenes_' \
                  + args.style + '.pkl', 'wb') as f:
        pickle.dump(dict_dest, f)

if __name__ == '__main__':
    main()
