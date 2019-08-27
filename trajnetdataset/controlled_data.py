import random
import numpy as np
import rvo2
import socialforce
import argparse
import os
SQUARE_LENGTH = 20


### Synthetic Data ####
### Initially: scenario3 - 60%, scenario2 - 20%, random - 10%, scenario1 - 10% ###  
#######################

def initialize(scenario, num_ped, sim=None):

    if scenario == 'random':
        return random_initialize(num_ped, sim)
    elif scenario == 'scenario_1':
        return scenario1_initialize(num_ped, sim)
    elif scenario == 'scenario_2':
        return scenario2_initialize(num_ped, sim)
    elif 'overfit' in scenario:
        return overfit_initialize(num_ped, sim)
    else:
        return scenario3_initialize(num_ped, sim)


def overfit_initialize(num_ped, sim):
    # initialize agents' starting and goal positions
    ## Time Varying Interaction (from left to right) + Noise 
    x = np.linspace(-10, 10, 8)
    positions = []
    goals = []
    speed = []
    num_ped = 8
    for i in range(8):
        random_number = random.uniform(-0.3, 0.3)
        # py = [-5, 5]
        py = [-7, 7]
        gy = [7, -7]
        for j in range(2):
            px = x[i] + random_number
            gx = x[i] + random_number
            py_ = py[j] + i * 5/9 * np.sign(j) + random.uniform(-0.5, 0.5)
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


def scenario1_initialize(num_ped, sim):
    # initialize agents' starting and goal positions
    # No time varying interaction
    x = np.linspace(-10, 10, int(np.around(num_ped/2+0.1)))
    positions = []
    goals = []
    speed = []
    for i in range(int(np.around(num_ped/2+0.1))):
        py = [-5, 5]
        gy = [7, -7]
        for j in range(2):
            px = x[i]+random.uniform(-0.5, 0.5)
            gx = x[i]+random.uniform(-0.5, 0.5)
            py_ = py[j] + random.uniform(-1, 1)
            gy_ = gy[j] + random.uniform(-1, 1)
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


def scenario2_initialize(num_ped, sim=None):
    # initialize agents' starting and goal positions
    pass


def scenario3_initialize(num_ped, sim=None):
    # initialize agents' starting and goal positions
    pass


def random_initialize(num_ped, sim=None):
    # initialize agents' starting and goal positions
    pass


def generate_orca_trajectory(scenario, num_ped, end_range=1):
    sim = rvo2.PyRVOSimulator(1 / 2.5, 3, 10, 1.5, 2, 0.4, 2)

    ##Initiliaze a scene
    trajectories, positions, goals, speed = initialize(scenario, num_ped, sim=sim)
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
                pref_vel = velocity / speed if speed > 1.2 else velocity
                sim.setAgentPrefVelocity(i, tuple(pref_vel.tolist()))
        count += 1
        done = all(reaching_goal)

    return trajectories


def generate_sf_trajectory(scenario, num_ped, end_range=1):

    ##Initiliaze a scene
    trajectories, positions, goals, speed = initialize(scenario, num_ped)

    initial_state = np.array([[positions[i][0], positions[i][1], speed[i][0], speed[i][1],
                               goals[i][0], goals[i][1]] for i in range(num_ped)])
    reaching_goal = [False] * num_ped
    done = False
    count = 0

    ##Simulate a scene
    while not done and count < 300:
        count += 1
        s = socialforce.Simulator(initial_state, delta_t=0.2)
        position = np.stack(s.step().state.copy())
        for i in range(len(initial_state)):
            if count % 2 == 0:
                trajectories[i].append((position[i, 0], position[i, 1]))
            # check if this agent reaches the goal
            if np.linalg.norm(position[i, :2] - np.array(goals[i])) < end_range:
                reaching_goal[i] = True
            else:
                initial_state[i, :2] = position[i, :2]

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


def generate_trajectory(simulator, scenario, num_ped):

    if simulator == 'orca':
        return generate_orca_trajectory(scenario=scenario, num_ped=num_ped)
    else:
        return generate_sf_trajectory(scenario=scenario, num_ped=num_ped)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--simulator', default='orca',
                        choices=('orca', 'social_force'))
    parser.add_argument('--simulation-type',  required=True, nargs='+')

    args = parser.parse_args()

    if not os.path.isdir('./data'):
        os.makedirs('./data')

    for simulation in args.simulation_type:
        print(simulation)

        ##Decide the number of scenes 
        if simulation == 'scenario_1':
            N = 100  
            # N = 80
        else:
            # N = 1500
            N = 50

        count = 0
        last_frame = -5
        for i in range(N):
            ## Print every 10th scene
            if (i+1) % 10 == 0:
                print(i)
            ##Decide number of people
            if simulation != 'scenario_1':
                num_ped = random.randint(20, 40)
            else:
                num_ped = 16

            if simulation == 'overfit_initialize':
                num_ped = 8

            ##Generate the scene
            trajectories = generate_trajectory(simulator=args.simulator, scenario=simulation,
                                               num_ped=num_ped)
            ##Trajectories = Trajectories of all agents

            ##Write the Scene to Txt
            ##################################################
            ##WHY '+5' -->  Just a gap needed.  DONE 
            ##Does it work if only 1 trajectory --->  Not currently 
            ##################################################
            last_frame = write_to_txt(trajectories, 'data/raw/controlled/' + args.simulator + '_traj_'
                                      + simulation + '.txt', count=count, frame=last_frame+5)
            count += num_ped


if __name__ == '__main__':
    main()