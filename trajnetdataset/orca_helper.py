import rvo2

def predict_all(input_paths, goals, n_predict=12):
    pred_length = n_predict
    fps = 100
    sampling_rate = fps / 2.5
    sim = rvo2.PyRVOSimulator(1/fps, 4, 10, 4, 5, 0.6, 1.5) ## (TrajNet++)

    # initialize
    trajectories = [[(p[0], p[1])] for p in input_paths[-1]]
    [sim.addAgent((p[0], p[1])) for p in input_paths[-1]]
    num_ped = len(trajectories)

    for i in range(num_ped):
        velocity = np.array((input_paths[-1][i][0] - input_paths[-3][i][0], input_paths[-1][i][1] - input_paths[-3][i][1]))
        velocity = velocity/0.8
        sim.setAgentVelocity(i, tuple(velocity.tolist()))
        velocity = np.array((goals[i][0] - input_paths[-1][i][0], goals[i][1] - input_paths[-1][i][1]))
        speed = np.linalg.norm(velocity)
        pref_vel = 1 * velocity / speed if speed > 1 else velocity
        sim.setAgentPrefVelocity(i, tuple(pref_vel.tolist()))

    reaching_goal_by_ped = [False] * num_ped
    count = 0
    end_range = 1.0
    done = False
    ##Simulate a scene
    while count < sampling_rate * pred_length + 1:
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

    # states = np.array(trajectories[0])
    # return states
    return trajectories