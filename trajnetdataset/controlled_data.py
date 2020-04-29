""" Generating Controlled data for pretraining collision avoidance """

import random
import argparse
import os
import itertools

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

import rvo2
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

def generate_square_crossing(num_ped, sim=None, square_width=4):
    positions = []
    goals = []
    speed = []
    agent_list = []

    for _ in range(num_ped):
        if random.uniform(0, 1) > 0.5:
            sign = -1
        else:
            sign = 1
        min_dist = 0.8
        while True:
            px = random.uniform(0, 1) * square_width * 0.5 * sign
            py = (random.uniform(0, 1) - 0.5) * square_width
            collide = False
            for agent in agent_list:
                if norm((px - agent[0], py - agent[1])) < min_dist:
                    collide = True
                    break
            if not collide:
                break
        while True:
            gx = random.uniform(0, 1) * square_width * 0.5 * -sign
            gy = (random.uniform(0, 1) - 0.5) * square_width
            collide = False
            for agent in agent_list:
                if norm((gx - agent[2], gy - agent[3])) < min_dist:
                    collide = True
                    break
            if not collide:
                break

        positions.append((px, py))
        goals.append((gx, gy))
        if sim is not None:
            sim.addAgent((px, py))
        velocity = np.array([gx - px, gy - py])
        magnitude = np.linalg.norm(velocity)
        init_vel = 1 * velocity / magnitude if magnitude > 1 else velocity
        speed.append([init_vel[0], init_vel[1]])

        agent_list.append([px, py, gx, gy])

    trajectories = [[positions[i]] for i in range(num_ped)]
    return trajectories, positions, goals, speed

def overfit_initialize(num_ped, sim=None):
    """ Scenario initialization """

    # initialize agents' starting and goal positions
    x = np.linspace(-15, 15, 4)
    positions = []
    goals = []
    speed = []
    for i in range(4):
        # random_number = random.uniform(-0.3, 0.3)
        py = [-10, 10]
        gy = [10, -10]
        for j in range(2):
            px = x[i] + np.random.normal(0, 0.3) * np.sign(j)
            gx = x[i] + np.random.normal(0, 0.1) * np.sign(j)
            py_ = py[j] + i * 16/9 * np.sign(j) + random.uniform(-0.5, 0.5)
            gy_ = gy[j] + random.uniform(-0.5, 0.5)
            positions.append((px, py_))
            goals.append((gx, gy_))
            if sim is not None:
                sim.addAgent((px, py_))

            rand_speed = random.uniform(0.8, 1.2)
            vx = 0
            vy = rand_speed * np.sign(gy[j] - py_)
            speed.append((vx, vy))

    trajectories = [[positions[i]] for i in range(num_ped)]
    return trajectories, positions, goals, speed

def overfit_initialize_circle(num_ped, sim=None, center=(0, 0), radius=10):
    positions = []
    goals = []
    speed = []
    step = (2 * np.pi) / num_ped
    radius = radius + np.random.uniform(-3, 3)
    for pos in range(num_ped):
        angle = pos * step + np.random.uniform(-0.01, 0.01)
        px = center[0] + radius * np.cos(angle)
        py_ = center[1] + radius * np.sin(angle)
        gx = center[0] + radius * np.cos(angle + np.pi)
        gy_ = center[1] + radius * np.sin(angle + np.pi)
        positions.append((px, py_))
        goals.append((gx, gy_))

        if sim is not None:
            sim.addAgent((px, py_))

        ## v1.0
        # rand_speed = random.uniform(0.8, 1.2)
        # vx = 0
        # vy = 0
        # speed.append((vx, vy))

        ## v2.0
        velocity = np.array([gx - px, gy_ - py_])
        magnitude = np.linalg.norm(velocity)
        init_vel = 1 * velocity / magnitude if magnitude > 1 else velocity
        speed.append([init_vel[0], init_vel[1]])

    trajectories = [[positions[i]] for i in range(num_ped)]
    return trajectories, positions, goals, speed

def generate_orca_trajectory(sim_scene, num_ped, min_dist=3, react_time=1.5, end_range=1.0, mode=None):
    """ Simulating Scenario using ORCA """
    ## Default: (1 / 60., 1.5, 5, 1.5, 2, 0.4, 2)
    sampling_rate = 1

    ##Initiliaze simulators & scenes
    if sim_scene == 'two_ped':
        sim = rvo2.PyRVOSimulator(1 / 2.5, min_dist, 10, react_time, 2, 0.4, 2)
        trajectories, _, goals, speed = overfit_initialize(num_ped, sim)

    ## Circle Overfit
    elif sim_scene == 'circle_overfit':
        sim = rvo2.PyRVOSimulator(1 / 2.5, 2, 10, 2, 2, 0.4, 1.2)
        trajectories, _, goals, speed = overfit_initialize_circle(num_ped, sim)

    ## Circle Crossing
    elif sim_scene == 'circle_crossing':
        fps = 60
        sampling_rate = fps / 2.5
        sim = rvo2.PyRVOSimulator(1/fps, 10, 10, 5, 5, 0.3, 1)
        if mode == 'trajnet':
            sim = rvo2.PyRVOSimulator(1/fps, 4, 10, 4, 5, 0.6, 1.5) ## (TrajNet++)
        trajectories, _, goals, speed = generate_circle_crossing(num_ped, sim, mode=mode)

    ## Square Crossing
    elif sim_scene == 'square_crossing':
        fps = 5
        sampling_rate = fps / 2.5
        sim = rvo2.PyRVOSimulator(1/fps, 10, 10, 5, 5, 0.3, 1)
        trajectories, _, goals, speed = generate_square_crossing(num_ped, sim)

    else:
        raise NotImplementedError

    # run
    done = False
    reaching_goal_by_ped = [False] * num_ped
    count = 0
    valid = True

    ##Simulate a scene
    while not done and count < 6000:
        sim.doStep()
        reaching_goal = []
        for i in range(num_ped):
            if count == 0:
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
        count += 1
        done = all(reaching_goal)

    if not done or not are_smoothes(trajectories):
        valid = False

    return trajectories, valid

def generate_sf_trajectory(sim_scene, num_ped, sf_params=[0.5, 2.1, 0.3], end_range=0.2):
    """ Simulating Scenario using SF """
    ## Default: (0.5, 2.1, 0.3)
    sampling_rate = 1

    ##Initiliaze simulators & scenes
    if sim_scene == 'two_ped':
        trajectories, positions, goals, speed = overfit_initialize(num_ped)

    ## Circle Overfit
    elif sim_scene == 'circle_overfit':
        trajectories, positions, goals, speed = overfit_initialize_circle(num_ped)

    ## Circle Crossing
    elif sim_scene == 'circle_crossing':
        fps = 10
        sampling_rate = fps / 2.5
        trajectories, positions, goals, speed = generate_circle_crossing(num_ped)

    ## Square Crossing
    elif sim_scene == 'square_crossing':
        fps = 10
        sampling_rate = fps / 2.5
        trajectories, positions, goals, speed = generate_square_crossing(num_ped)

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

def write_to_txt(trajectories, path, count, frame):
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--simulator', default='orca',
                        choices=('orca', 'social_force'))
    parser.add_argument('--simulation_scene', default='circle_crossing',
                        choices=('circle_crossing', 'square_crossing',
                                 'circle_overfit', 'two_ped'))
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

    if args.simulation_scene == 'two_ped':
        num_ped = 2

    if not os.path.isdir('./data'):
        os.makedirs('./data')

    if args.style is not None and args.simulator == 'orca':
        ## ORCA Params: For Two Ped / Different Styles
        # # train_params (min_dist, react_time) = [[2, 2.5], [3, 1], [3, 2], [4, 2]]
        # ## [3, 1] is close
        # ## [2, 2.5], [3, 2] is medium
        # ## [4, 2] is far
        dict_params = {}
        dict_params['close'] = [3, 1]
        dict_params['medium1'] = [2, 2.5]
        dict_params['medium2'] = [3, 2]
        dict_params['far'] = [4, 2]
        [min_dist, react_time] = dict_params[args.proximity]
        print("min_dist, time_react:", min_dist, react_time)
    else:
        min_dist, react_time = 1.5, 1.5
        args.style = ''

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

    for i in range(num_scenes):
        if mode == 'trajnet':
            num_ped = random.choice([5, 6, 7, 8]) ## TrajNet++
        ## Print every 10th scene
        if (i+1) % 10 == 0:
            print(i)

        ##Generate scenes
        if args.simulator == 'orca':
            trajectories, valid = generate_orca_trajectory(sim_scene=args.simulation_scene,
                                                       num_ped=num_ped,
                                                       min_dist=min_dist,
                                                       react_time=react_time,
                                                       mode=mode)
        elif args.simulator == 'social_force':
            trajectories, valid = generate_sf_trajectory(sim_scene=args.simulation_scene,
                                                     num_ped=num_ped,
                                                     sf_params=[0.5, 1.0, 0.1])
        else:
            raise NotImplementedError

        ## Visualizing scenes
        # print("VALID : ", valid)
        if not valid:
            viz(trajectories, mode=mode)

        ## Write
        if valid:
            last_frame = write_to_txt(trajectories, output_file,
                                      count=count, frame=last_frame+5)

        count += num_ped

if __name__ == '__main__':
    main()
