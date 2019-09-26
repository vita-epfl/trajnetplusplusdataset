import random
import numpy as np
import rvo2
import argparse
import os
import matplotlib.pyplot as plt

### Controlled Data Generation ####
def overfit_initialize(num_ped, sim):
    # initialize agents' starting and goal positions
    ## Time Varying Interaction (from left to right) + Noise 
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

def generate_orca_trajectory(num_ped, min_dist=3, react_time=1.5, end_range=1.0):
    sim = rvo2.PyRVOSimulator(1 / 2.5, min_dist, 10, react_time, 2, 0.4, 2)
    #1.5

    ##Initiliaze a scene
    trajectories, positions, goals, speed = overfit_initialize(num_ped, sim)
    done = False
    reaching_goal_by_ped = [False] * num_ped
    count = 0

    ##Simulate a scene
    while not done and count < 150:
        sim.doStep()
        reaching_goal = []
        for i in range(num_ped):
            if count == 0:
                trajectories[i].pop(0)
            position = sim.getAgentPosition(i)

            ## Append only if Goal not reached
            if not reaching_goal_by_ped[i]:
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
                pref_vel = velocity / speed if speed > 1 else velocity
                sim.setAgentPrefVelocity(i, tuple(pref_vel.tolist()))
        count += 1
        done = all(reaching_goal)

    return trajectories

### Write Trajectories to the text file
def write_to_txt(trajectories, path, count, frame):
    last_frame = 0
    with open(path, 'a') as fo:
        track_data = []
        for i in range(len(trajectories)):
            for t in range(len(trajectories[i])):

                track_data.append('{}, {}, {}, {}'.format(t+frame, count+i,
                                               trajectories[i][t][0], trajectories[i][t][1]))
                if t == len(trajectories[i])-1 and t+frame > last_frame:
                    last_frame = t+frame

        for track in track_data:
            fo.write(track)
            fo.write('\n')

    return last_frame

def viz(trajectories):
    for i in range(len(trajectories)):
        trajectory = np.array(trajectories[i])
        plt.plot(trajectory[:, 0], trajectory[:, 1])

    plt.xlim(-16, 16)
    plt.show()
    plt.close()
    return

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--simulator', default='orca',
                        choices=('orca', 'social_force'))
    parser.add_argument('--simulation_type', required=True)
    parser.add_argument('--test', default=False)

    args = parser.parse_args()

    if not os.path.isdir('./data'):
        os.makedirs('./data')

    # # train_params = [[2, 2.5], [3, 1], [3, 2], [4, 2]]
    # ## [3, 1] is close 
    # ## [2, 2.5], [3, 2] is medium
    # ## [4, 2] is far
    if args.simulation_type == 'close':
        params = [[3, 1]]
    elif args.simulation_type == 'medium1':
        params = [[2, 2.5]]
    elif args.simulation_type == 'medium2':
        params = [[3, 2]]
    elif args.simulation_type == 'far':
        params = [[4, 2]]
    else:
        raise ValueError

    for min_dist, react_time in params:
        print("min_dist, time_react:", min_dist, react_time) 
        ##Decide the number of scenes 
        if not args.test:
            N = 100 
        else:
            N = 10
        ##Decide number of people
        num_ped = 8

        count = 0   
        last_frame = -5
        for i in range(N):
            ## Print every 10th scene
            if (i+1) % 10 == 0:
                print(i)

            ##Generate the scene
            trajectories = generate_orca_trajectory(num_ped=num_ped, min_dist=min_dist, react_time=react_time)
            # viz(trajectories)
            # ##Write the Scene to Txt
            if not args.test:
                last_frame = write_to_txt(trajectories, 'data/raw/controlled/' + args.simulator + '_traj_'
                                          + args.simulation_type + '.txt', count=count, frame=last_frame+5)
            else:
                last_frame = write_to_txt(trajectories, 'data/raw/controlled/test_' + args.simulator + '_traj_'
                                          + args.simulation_type + '.txt', count=count, frame=last_frame+5)
            count += num_ped


if __name__ == '__main__':
    main()