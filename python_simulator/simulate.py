import numpy as np
from scipy.special import expit


class Simulation:
    def __init__(self, step_size, evaluation_params):
        self.width = evaluation_params['screen_width']  # [-20, 20]
        self.step_size = step_size  # how fast things are happening in the simulation
        # self.trials = self.create_trials(evaluation_params['velocities'], evaluation_params['impacts'])
        self.trials = self.create_trials(evaluation_params['velocities'], evaluation_params['impacts'],
                                         evaluation_params['tg_start_range'], evaluation_params['tg_start_variants'])
        distance = (self.width[1]-self.width[0]) * evaluation_params['n_turns']  # total distance travelled by target
        # simulation length depends on target velocity
        self.sim_length = [int(distance/abs(trial[0])/self.step_size) for trial in self.trials]
        # self.sim_length = [500] * len(self.trials)
        self.condition = evaluation_params['condition']  # is it a sound condition?
        # the period of time at the beginning of the trial in which the target stays still
        self.start_period = evaluation_params['start_period']
        self.initial_state = evaluation_params['initial_state']
        self.velocity_control = evaluation_params['velocity_control']

    # @staticmethod
    # def create_trials(velocities, impacts):
    #     """
    #     Create a list of trials the environment will run.
    #     :return:
    #     """
    #     trials = [(x, y) for x in velocities for y in impacts]
    #     return trials

    @staticmethod
    def create_trials(velocities, impacts, start_range, size):
        """
        Create a list of trials the environment will run.
        :return:
        """
        if start_range[1]-start_range[0] == 0:
            trials = [(x, y, start_range[0]) for x in velocities for y in impacts]
        else:
            left_positions = np.random.choice(np.arange(start_range[0], 1), size, replace=False)
            right_positions = np.random.choice(np.arange(0, start_range[1] + 1), size, replace=False)
            target_positions = np.concatenate((left_positions, right_positions))
            trials = [(x, y, z) for x in velocities for y in impacts for z in target_positions]
        return trials

    def run_trials(self, agent, trials, savedata=False):
        """
        An evaluation function that accepts an agent and returns a real number representing
        the performance of that parameter vector on the task. Here the task is the Knoblich and Jordan task.

        :param agent: an agent with a CTRNN brain and particular anatomy
        :param trials: a list of trials to perform
        :param savedata: should all the trial data be saved
        :return: fitness
        """

        n_trials = len(trials)
        trial_data = dict()
        trial_data['fitness'] = []
        trial_data['target_pos'] = [None] * n_trials
        trial_data['tracker_pos'] = [None] * n_trials
        trial_data['tracker_v'] = [None] * n_trials
        trial_data['keypress'] = [None] * n_trials

        if savedata:
            trial_data['brain_state'] = [None] * n_trials
            trial_data['derivatives'] = [None] * n_trials
            trial_data['input'] = [None] * n_trials
            trial_data['output'] = [None] * n_trials
            trial_data['button_state'] = [None] * n_trials

        for i in range(n_trials):
            # create target and tracker
            # target = Target(trials[i][0], self.step_size, 0)
            target = Target(trials[i][0], self.step_size, trials[i][2])

            tracker = None
            if self.velocity_control == "buttons":
                tracker = Tracker(trials[i][1], self.step_size, self.condition)
            elif self.velocity_control == "direct":
                tracker = DirectTracker(None, self.step_size, self.condition)

            # set initial state in specified range
            agent.brain.randomize_state(self.initial_state)
            agent.initialize_buttons()

            trial_data['target_pos'][i] = np.zeros((self.sim_length[i] + self.start_period, 1))
            trial_data['tracker_pos'][i] = np.zeros((self.sim_length[i] + self.start_period, 1))
            trial_data['tracker_v'][i] = np.zeros((self.sim_length[i] + self.start_period, 1))
            trial_data['keypress'][i] = np.zeros((self.sim_length[i] + self.start_period, 2))

            if savedata:
                trial_data['brain_state'][i] = np.zeros((self.sim_length[i] + self.start_period, agent.brain.N))
                trial_data['derivatives'][i] = np.zeros((self.sim_length[i] + self.start_period, agent.brain.N))
                trial_data['input'][i] = np.zeros((self.sim_length[i] + self.start_period, agent.brain.N))
                trial_data['output'][i] = np.zeros((self.sim_length[i] + self.start_period, 2))
                trial_data['button_state'][i] = np.zeros((self.sim_length[i] + self.start_period, 2))

            if self.start_period > 0:
                # don't move the target and don't allow tracker to move
                for j in range(self.start_period):
                    agent.visual_input(tracker.position, target.position)
                    agent.brain.euler_step()
                    # activation, motor_activity = agent.motor_output()
                    # activation = agent.motor_output()
                    # tracker.accelerate(activation)

                    trial_data['target_pos'][i][j] = target.position
                    trial_data['tracker_pos'][i][j] = tracker.position
                    trial_data['tracker_v'][i][j] = tracker.velocity
                    # trial_data['keypress'][i][j] = activation

                    if savedata:
                        trial_data['brain_state'][i][j] = agent.brain.Y
                        trial_data['derivatives'][i][j] = agent.brain.dy_dt
                        trial_data['input'][i][j] = agent.brain.I
                        # trial_data['output'][i][j] = motor_activity
                        trial_data['button_state'][i][j] = agent.button_state

            for j in range(self.start_period, self.sim_length[i] + self.start_period):

                # 1) Target movement
                target.movement(self.width)
                # target.movement([self.width[0] + 10, self.width[1] - 10])

                # 2) Agent sees
                agent.visual_input(tracker.position, target.position)

                # 3) Agents moves
                sound_output = tracker.movement(self.width)

                # 4) Agent hears
                if self.condition == 'sound':
                    agent.auditory_input(sound_output)

                trial_data['target_pos'][i][j] = target.position
                trial_data['tracker_pos'][i][j] = tracker.position
                trial_data['tracker_v'][i][j] = tracker.velocity

                if savedata:
                    trial_data['brain_state'][i][j] = agent.brain.Y
                    trial_data['derivatives'][i][j] = agent.brain.dy_dt

                # 5) Update agent's neural system
                agent.brain.euler_step()

                # 6) Agent reacts
                activation, neuron_output = agent.motor_output()
                # activation = agent.motor_output()
                tracker.accelerate(activation)
                # this will save -1 or 1 for button-controlling agents
                # but left and right velocities for direct velocity control agent
                trial_data['keypress'][i][j] = activation

                if savedata:

                    trial_data['input'][i][j] = agent.brain.I
                    trial_data['output'][i][j] = neuron_output
                    trial_data['button_state'][i][j] = agent.button_state

            # 6) Fitness tacking:
            fitness = 1 - (np.sum(np.abs(trial_data['target_pos'][i] - trial_data['tracker_pos'][i])) /
                           (2 * self.width[1] * (self.sim_length[i] + self.start_period)))
            # penalty for not moving in the trial not counting the delay period
            penalty = list(trial_data['tracker_v'][i][self.start_period:]).count(0) / (self.sim_length[i])
            # if penalty decreases the score below 0, set it to 0
            overall_fitness = np.clip(fitness - penalty, 0, 1)
            trial_data['fitness'].append(overall_fitness)

            # trial_data['fitness'].append(np.mean(trial_data['keypress'][i]))

            # cap_distance = 10
            # total_dist = np.abs(trial_data['target_pos'][i] - trial_data['tracker_pos'][i])
            # scores = np.clip(-1/cap_distance * total_dist + 1, 0, 1)
            # trial_data['fitness'].append(np.mean(scores))
            # scores.sort(reverse=True)
            # trial_data['fitness'].append(np.mean(weighted_scores))

        return trial_data

    def run_joint_trials(self, agent1, agent2, trials, savedata=False):
        """
        An evaluation function that accepts two agents and returns a real number representing
        the performance of these agents on the task.
        Here the task is the Knoblich and Jordan task, the joint action version.

        :param agent1: an agent with a CTRNN brain and particular anatomy, controls only left button
        :param agent2: an agent with a CTRNN brain and particular anatomy, controls only right button
        :param trials: a list of trials to perform
        :param savedata: should all the trial data be saved
        :return: fitness
        """

        if self.width != [-20, 20]:
            agent1.max_dist = 60
            agent2.max_dist = 60
            agent1.visual_scale = agent1.max_dist / 10
            agent2.visual_scale = agent2.max_dist / 10
            agent1.screen_width = self.width
            agent2.screen_width = self.width

        n_trials = len(trials)
        trial_data = dict()
        trial_data['fitness'] = []
        trial_data['target_pos'] = [None] * n_trials
        trial_data['tracker_pos'] = [None] * n_trials
        trial_data['tracker_v'] = [None] * n_trials
        trial_data['keypress'] = [None] * n_trials

        if savedata:
            trial_data['brain_state_a1'] = [None] * n_trials
            trial_data['derivatives_a1'] = [None] * n_trials
            trial_data['input_a1'] = [None] * n_trials
            trial_data['output_a1'] = [None] * n_trials
            trial_data['button_state_a1'] = [None] * n_trials

            trial_data['brain_state_a2'] = [None] * n_trials
            trial_data['derivatives_a2'] = [None] * n_trials
            trial_data['input_a2'] = [None] * n_trials
            trial_data['output_a2'] = [None] * n_trials
            trial_data['button_state_a2'] = [None] * n_trials

        for i in range(n_trials):
            # create target and tracker
            # target = Target(trials[i][0], self.step_size, 0)
            target = Target(trials[i][0], self.step_size, trials[i][2])

            if self.velocity_control == "buttons":
                tracker = Tracker(trials[i][1], self.step_size, self.condition)
            elif self.velocity_control == "direct":
                tracker = DirectTracker(None, self.step_size, self.condition)

            # set initial state in specified range
            agent1.brain.randomize_state(self.initial_state)
            agent1.initialize_buttons()

            agent2.brain.randomize_state(self.initial_state)
            agent2.initialize_buttons()

            trial_data['target_pos'][i] = np.zeros((self.sim_length[i] + self.start_period, 1))
            trial_data['tracker_pos'][i] = np.zeros((self.sim_length[i] + self.start_period, 1))
            trial_data['tracker_v'][i] = np.zeros((self.sim_length[i] + self.start_period, 1))
            trial_data['keypress'][i] = np.zeros((self.sim_length[i] + self.start_period, 2))

            if savedata:
                trial_data['brain_state_a1'][i] = np.zeros((self.sim_length[i] + self.start_period, agent1.brain.N))
                trial_data['derivatives_a1'][i] = np.zeros((self.sim_length[i] + self.start_period, agent1.brain.N))
                trial_data['input_a1'][i] = np.zeros((self.sim_length[i] + self.start_period, agent1.brain.N))
                trial_data['output_a1'][i] = np.zeros((self.sim_length[i] + self.start_period, agent1.brain.N))
                trial_data['button_state_a1'][i] = np.zeros((self.sim_length[i] + self.start_period, 2))

                trial_data['brain_state_a2'][i] = np.zeros((self.sim_length[i] + self.start_period, agent2.brain.N))
                trial_data['derivatives_a2'][i] = np.zeros((self.sim_length[i] + self.start_period, agent2.brain.N))
                trial_data['input_a2'][i] = np.zeros((self.sim_length[i] + self.start_period, agent2.brain.N))
                trial_data['output_a2'][i] = np.zeros((self.sim_length[i] + self.start_period, agent1.brain.N))
                trial_data['button_state_a2'][i] = np.zeros((self.sim_length[i] + self.start_period, 2))

            if self.start_period > 0:
                # don't move the target and don't allow tracker to move
                for j in range(self.start_period):
                    agent1.visual_input(tracker.position, target.position)
                    agent1.brain.euler_step()
                    agent2.visual_input(tracker.position, target.position)
                    agent2.brain.euler_step()

                    # activation, motor_activity = agent.motor_output()
                    # activation = agent.motor_output()
                    # tracker.accelerate(activation)

                    trial_data['target_pos'][i][j] = target.position
                    trial_data['tracker_pos'][i][j] = tracker.position
                    trial_data['tracker_v'][i][j] = tracker.velocity
                    # trial_data['keypress'][i][j] = activation

                    if savedata:
                        trial_data['brain_state_a1'][i][j] = agent1.brain.Y
                        trial_data['derivatives_a1'][i][j] = agent1.brain.dy_dt
                        trial_data['input_a1'][i][j] = agent1.brain.I
                        trial_data['brain_state_a2'][i][j] = agent2.brain.Y
                        trial_data['derivatives_a2'][i][j] = agent2.brain.dy_dt
                        trial_data['input_a2'][i][j] = agent2.brain.I

                        trial_data['output_a1'][i][j] = expit(agent1.brain.Y + agent1.brain.Theta)
                        trial_data['output_a2'][i][j] = expit(agent2.brain.Y + agent2.brain.Theta)
                        trial_data['button_state_a1'][i][j] = agent1.button_state
                        trial_data['button_state_a2'][i][j] = agent2.button_state

            for j in range(self.start_period, self.sim_length[i] + self.start_period):

                # 1) Target movement
                target.movement(self.width)
                # target.movement([self.width[0] + 10, self.width[1] - 10])

                # 2) Agent sees
                agent1.visual_input(tracker.position, target.position)
                agent2.visual_input(tracker.position, target.position)

                # 3) Agents moves
                sound_output = tracker.movement(self.width)

                # 4) Agent hears
                if self.condition == 'sound':
                    agent1.auditory_input(sound_output)
                    agent2.auditory_input(sound_output)

                trial_data['target_pos'][i][j] = target.position
                trial_data['tracker_pos'][i][j] = tracker.position
                trial_data['tracker_v'][i][j] = tracker.velocity

                if savedata:
                    trial_data['brain_state_a1'][i][j] = agent1.brain.Y
                    trial_data['brain_state_a2'][i][j] = agent2.brain.Y
                    trial_data['derivatives_a1'][i][j] = agent1.brain.dy_dt
                    trial_data['derivatives_a2'][i][j] = agent2.brain.dy_dt

                # 5) Update agent's neural system
                agent1.brain.euler_step()
                agent2.brain.euler_step()

                # 6) Agent reacts
                activation1, neuron_output1 = agent1.motor_output()
                activation2, neuron_output2 = agent2.motor_output()
                activation_left = activation1[0]  # take only left activation from "left" agent
                activation_right = activation2[1]  # take only right activation from "right" agent

                activation = [activation_left, activation_right]
                tracker.accelerate(activation)
                # this will save -1 or 1 for button-controlling agents
                # but left and right velocities for direct velocity control agent
                trial_data['keypress'][i][j] = activation

                if savedata:
                    trial_data['input_a1'][i][j] = agent1.brain.I
                    trial_data['output_a1'][i][j] = expit(agent1.brain.Y + agent1.brain.Theta)
                    trial_data['button_state_a1'][i][j] = agent1.button_state
                    trial_data['input_a2'][i][j] = agent2.brain.I
                    trial_data['output_a2'][i][j] = expit(agent2.brain.Y + agent2.brain.Theta)
                    trial_data['button_state_a2'][i][j] = agent2.button_state

            # 6) Fitness tacking:
            fitness = 1 - (np.sum(np.abs(trial_data['target_pos'][i] - trial_data['tracker_pos'][i])) /
                           (2 * self.width[1] * (self.sim_length[i] + self.start_period)))
            # penalty for not moving in the trial not counting the delay period
            penalty = list(trial_data['tracker_v'][i][self.start_period:]).count(0) / (self.sim_length[i])
            # if penalty decreases the score below 0, set it to 0
            overall_fitness = np.clip(fitness - penalty, 0, 1)
            trial_data['fitness'].append(overall_fitness)

            # trial_data['fitness'].append(np.mean(trial_data['keypress'][i]))

            # cap_distance = 10
            # total_dist = np.abs(trial_data['target_pos'][i] - trial_data['tracker_pos'][i])
            # scores = np.clip(-1/cap_distance * total_dist + 1, 0, 1)
            # trial_data['fitness'].append(np.mean(scores))
            # scores.sort(reverse=True)
            # trial_data['fitness'].append(np.mean(weighted_scores))

        return trial_data

    def run_playback_trials(self, agent, playback_data, trials, active_agent, savedata=False):
        """
        A version of the trials in which one agent's activity is not produced real-time but replayed from previous
        run of the trials.
        :param agent: an agent with a CTRNN brain and particular anatomy, controls only left/right button
        :param playback_data: played back data
        :param trials: a list of trials to perform
        :param active_agent: which agent is active
        :param savedata: should all the trial data be saved
        :return: fitness
        """

        if self.width != [-20, 20]:
            agent.max_dist = 60
            agent.visual_scale = agent.max_dist / 10
            agent.screen_width = self.width

        n_trials = len(trials)
        trial_data = dict()
        trial_data['fitness'] = []
        trial_data['target_pos'] = [None] * n_trials
        trial_data['tracker_pos'] = [None] * n_trials
        trial_data['tracker_v'] = [None] * n_trials
        trial_data['keypress'] = [None] * n_trials

        if savedata:
            trial_data['brain_state_a1'] = [None] * n_trials
            trial_data['derivatives_a1'] = [None] * n_trials
            trial_data['input_a1'] = [None] * n_trials
            trial_data['output_a1'] = [None] * n_trials
            trial_data['button_state_a1'] = [None] * n_trials

        for i in range(n_trials):
            # create target and tracker
            # target = Target(trials[i][0], self.step_size, 0)
            target = Target(trials[i][0], self.step_size, trials[i][2])

            if self.velocity_control == "buttons":
                tracker = Tracker(trials[i][1], self.step_size, self.condition)
            elif self.velocity_control == "direct":
                tracker = DirectTracker(None, self.step_size, self.condition)

            # set initial state in specified range
            agent.brain.randomize_state(self.initial_state)
            agent.initialize_buttons()

            trial_data['target_pos'][i] = np.zeros((self.sim_length[i] + self.start_period, 1))
            trial_data['tracker_pos'][i] = np.zeros((self.sim_length[i] + self.start_period, 1))
            trial_data['tracker_v'][i] = np.zeros((self.sim_length[i] + self.start_period, 1))
            trial_data['keypress'][i] = np.zeros((self.sim_length[i] + self.start_period, 2))

            if savedata:
                trial_data['brain_state_a1'][i] = np.zeros((self.sim_length[i] + self.start_period, agent.brain.N))
                trial_data['derivatives_a1'][i] = np.zeros((self.sim_length[i] + self.start_period, agent.brain.N))
                trial_data['input_a1'][i] = np.zeros((self.sim_length[i] + self.start_period, agent.brain.N))
                trial_data['output_a1'][i] = np.zeros((self.sim_length[i] + self.start_period, agent.brain.N))
                trial_data['button_state_a1'][i] = np.zeros((self.sim_length[i] + self.start_period, 2))

            if self.start_period > 0:
                # don't move the target and don't allow tracker to move
                for j in range(self.start_period):
                    agent.visual_input(tracker.position, target.position)
                    agent.brain.euler_step()

                    # activation, motor_activity = agent.motor_output()
                    # activation = agent.motor_output()
                    # tracker.accelerate(activation)

                    trial_data['target_pos'][i][j] = target.position
                    trial_data['tracker_pos'][i][j] = tracker.position
                    trial_data['tracker_v'][i][j] = tracker.velocity
                    # trial_data['keypress'][i][j] = activation

                    if savedata:
                        trial_data['brain_state_a1'][i][j] = agent.brain.Y
                        trial_data['derivatives_a1'][i][j] = agent.brain.dy_dt
                        trial_data['input_a1'][i][j] = agent.brain.I

                        trial_data['output_a1'][i][j] = expit(agent.brain.Y + agent.brain.Theta)
                        trial_data['button_state_a1'][i][j] = agent.button_state

            for j in range(self.start_period, self.sim_length[i] + self.start_period):

                # 1) Target movement
                target.movement(self.width)
                # target.movement([self.width[0] + 10, self.width[1] - 10])

                # 2) Agent sees
                agent.visual_input(tracker.position, target.position)

                # 3) Agents moves
                sound_output = tracker.movement(self.width)

                # 4) Agent hears
                if self.condition == 'sound':
                    agent.auditory_input(sound_output)

                trial_data['target_pos'][i][j] = target.position
                trial_data['tracker_pos'][i][j] = tracker.position
                trial_data['tracker_v'][i][j] = tracker.velocity

                if savedata:
                    trial_data['brain_state_a1'][i][j] = agent.brain.Y
                    trial_data['derivatives_a1'][i][j] = agent.brain.dy_dt

                # 5) Update agent's neural system
                agent.brain.euler_step()

                # 6) Agent reacts
                activation1, neuron_output1 = agent.motor_output()  # active agent output
                if active_agent == 'left':
                    activation_left = activation1[0]  # take only left activation from "left" agent
                    activation_right = playback_data['keypress'][i][j][1]
                else:  # active_agent == 'right'
                    activation_left = playback_data['keypress'][i][j][0]
                    activation_right = activation1[1]  # take only right activation from "right" agent

                activation = [activation_left, activation_right]
                tracker.accelerate(activation)
                # this will save -1 or 1 for button-controlling agents
                # but left and right velocities for direct velocity control agent
                trial_data['keypress'][i][j] = activation

                if savedata:
                    trial_data['input_a1'][i][j] = agent.brain.I
                    trial_data['output_a1'][i][j] = expit(agent.brain.Y + agent.brain.Theta)
                    trial_data['button_state_a1'][i][j] = agent.button_state

            # 6) Fitness tacking:
            fitness = 1 - (np.sum(np.abs(trial_data['target_pos'][i] - trial_data['tracker_pos'][i])) /
                           (2 * self.width[1] * (self.sim_length[i] + self.start_period)))
            # penalty for not moving in the trial not counting the delay period
            penalty = list(trial_data['tracker_v'][i][self.start_period:]).count(0) / (self.sim_length[i])
            # if penalty decreases the score below 0, set it to 0
            overall_fitness = np.clip(fitness - penalty, 0, 1)
            trial_data['fitness'].append(overall_fitness)

            # trial_data['fitness'].append(np.mean(trial_data['keypress'][i]))

            # cap_distance = 10
            # total_dist = np.abs(trial_data['target_pos'][i] - trial_data['tracker_pos'][i])
            # scores = np.clip(-1/cap_distance * total_dist + 1, 0, 1)
            # trial_data['fitness'].append(np.mean(scores))
            # scores.sort(reverse=True)
            # trial_data['fitness'].append(np.mean(weighted_scores))

        return trial_data


class LesionedSimulation:
    def __init__(self, step_size, evaluation_params):
        self.width = evaluation_params['screen_width']  # [-20, 20]
        self.step_size = step_size  # how fast things are happening in the simulation
        # self.trials = self.create_trials(evaluation_params['velocities'], evaluation_params['impacts'])
        self.trials = self.create_trials(evaluation_params['velocities'], evaluation_params['impacts'],
                                         evaluation_params['tg_start_range'], evaluation_params['tg_start_variants'])
        distance = (self.width[1]-self.width[0]) * evaluation_params['n_turns']  # total distance travelled by target
        # simulation length depends on target velocity
        self.sim_length = [int(distance/abs(trial[0])/self.step_size) for trial in self.trials]
        # self.sim_length = [500] * len(self.trials)
        self.condition = evaluation_params['condition']  # is it a sound condition?
        # the period of time at the beginning of the trial in which the target stays still
        self.start_period = evaluation_params['start_period']
        self.initial_state = evaluation_params['initial_state']
        self.velocity_control = evaluation_params['velocity_control']

    @staticmethod
    def create_trials(velocities, impacts, start_range, size):
        """
        Create a list of trials the environment will run.
        :return:
        """
        if start_range[1]-start_range[0] == 0:
            trials = [(x, y, start_range[0]) for x in velocities for y in impacts]
        else:
            left_positions = np.random.choice(np.arange(start_range[0], 1), size, replace=False)
            right_positions = np.random.choice(np.arange(0, start_range[1] + 1), size, replace=False)
            target_positions = np.concatenate((left_positions, right_positions))
            trials = [(x, y, z) for x in velocities for y in impacts for z in target_positions]
        return trials

    def run_trials(self, agent, trials, lesion_point, lesion_type, savedata=False):
        """
        An evaluation function that accepts an agent and returns a real number representing
        the performance of that parameter vector on the task. Here the task is the Knoblich and Jordan task.

        :param agent: an agent with a CTRNN brain and particular anatomy
        :param trials: a list of trials to perform
        :param savedata: should all the trial data be saved
        :return: fitness
        """

        n_trials = len(trials)
        trial_data = dict()
        trial_data['fitness'] = []
        trial_data['target_pos'] = [None] * n_trials
        trial_data['tracker_pos'] = [None] * n_trials
        trial_data['tracker_v'] = [None] * n_trials
        trial_data['keypress'] = [None] * n_trials

        if savedata:
            trial_data['brain_state'] = [None] * n_trials
            trial_data['derivatives'] = [None] * n_trials
            trial_data['input'] = [None] * n_trials
            trial_data['output'] = [None] * n_trials
            trial_data['button_state'] = [None] * n_trials

        for i in range(n_trials):
            # create target and tracker
            # target = Target(trials[i][0], self.step_size, 0)
            target = Target(trials[i][0], self.step_size, trials[i][2])

            if self.velocity_control == "buttons":
                tracker = Tracker(trials[i][1], self.step_size, self.condition)
            elif self.velocity_control == "direct":
                tracker = DirectTracker(None, self.step_size, self.condition)

            # set initial state in specified range
            agent.brain.randomize_state(self.initial_state)
            agent.initialize_buttons()

            trial_data['target_pos'][i] = np.zeros((self.sim_length[i] + self.start_period, 1))
            trial_data['tracker_pos'][i] = np.zeros((self.sim_length[i] + self.start_period, 1))
            trial_data['tracker_v'][i] = np.zeros((self.sim_length[i] + self.start_period, 1))
            trial_data['keypress'][i] = np.zeros((self.sim_length[i] + self.start_period, 2))

            if savedata:
                trial_data['brain_state'][i] = np.zeros((self.sim_length[i] + self.start_period, agent.brain.N))
                trial_data['derivatives'][i] = np.zeros((self.sim_length[i] + self.start_period, agent.brain.N))
                trial_data['input'][i] = np.zeros((self.sim_length[i] + self.start_period, agent.brain.N))
                trial_data['output'][i] = np.zeros((self.sim_length[i] + self.start_period, 2))
                trial_data['button_state'][i] = np.zeros((self.sim_length[i] + self.start_period, 2))

            if self.start_period > 0:
                # don't move the target and don't allow tracker to move
                for j in range(self.start_period):
                    agent.visual_input(tracker.position, target.position)
                    agent.brain.euler_step()
                    # activation, motor_activity = agent.motor_output()
                    # activation = agent.motor_output()
                    # tracker.accelerate(activation)

                    trial_data['target_pos'][i][j] = target.position
                    trial_data['tracker_pos'][i][j] = tracker.position
                    trial_data['tracker_v'][i][j] = tracker.velocity
                    # trial_data['keypress'][i][j] = activation

                    if savedata:
                        trial_data['brain_state'][i][j] = agent.brain.Y
                        trial_data['derivatives'][i][j] = agent.brain.dy_dt
                        trial_data['input'][i][j] = agent.brain.I
                        # trial_data['output'][i][j] = motor_activity
                        trial_data['button_state'][i][j] = agent.button_state

            for j in range(self.start_period, self.sim_length[i] + self.start_period):

                # 1) Target movement
                target.movement(self.width)
                # target.movement([self.width[0] + 10, self.width[1] - 10])

                # 2) Agent sees
                if (lesion_type == "visual_border" or lesion_type == "visual_target") and \
                                j > (self.sim_length[i]/lesion_point):
                    agent.lesioned_visual_input(tracker.position, target.position, lesion_type)
                else:
                    agent.visual_input(tracker.position, target.position)

                # 3) Agents moves
                sound_output = tracker.movement(self.width)

                # 4) Agent hears
                if self.condition == 'sound':
                    if lesion_type == "auditory" and j > (self.sim_length[i]/lesion_point):
                        agent.auditory_input([0, 0])
                    else:
                        agent.auditory_input(sound_output)

                trial_data['target_pos'][i][j] = target.position
                trial_data['tracker_pos'][i][j] = tracker.position
                trial_data['tracker_v'][i][j] = tracker.velocity

                if savedata:
                    trial_data['brain_state'][i][j] = agent.brain.Y
                    trial_data['derivatives'][i][j] = agent.brain.dy_dt

                # 5) Update agent's neural system
                agent.brain.euler_step()

                # 6) Agent reacts
                activation, neuron_output = agent.motor_output()
                # activation = agent.motor_output()
                tracker.accelerate(activation)
                # this will save -1 or 1 for button-controlling agents
                # but left and right velocities for direct velocity control agent
                trial_data['keypress'][i][j] = activation

                if savedata:

                    trial_data['input'][i][j] = agent.brain.I
                    trial_data['output'][i][j] = neuron_output
                    trial_data['button_state'][i][j] = agent.button_state

            # 6) Fitness tacking:
            fitness = 1 - (np.sum(np.abs(trial_data['target_pos'][i] - trial_data['tracker_pos'][i])) /
                           (2 * self.width[1] * (self.sim_length[i] + self.start_period)))
            # penalty for not moving in the trial not counting the delay period
            penalty = list(trial_data['tracker_v'][i][self.start_period:]).count(0) / (self.sim_length[i])
            # if penalty decreases the score below 0, set it to 0
            overall_fitness = np.clip(fitness - penalty, 0, 1)
            trial_data['fitness'].append(overall_fitness)

        return trial_data

    def run_joint_trials(self, agent1, agent2, trials, lesion_point, lesion_type, savedata=False):
        """
        An evaluation function that accepts two agents and returns a real number representing
        the performance of these agents on the task.
        Here the task is the Knoblich and Jordan task, the joint action version.

        :param agent1: an agent with a CTRNN brain and particular anatomy, controls only left button
        :param agent2: an agent with a CTRNN brain and particular anatomy, controls only right button
        :param trials: a list of trials to perform
        :param savedata: should all the trial data be saved
        :return: fitness
        """

        n_trials = len(trials)
        trial_data = dict()
        trial_data['fitness'] = []
        trial_data['target_pos'] = [None] * n_trials
        trial_data['tracker_pos'] = [None] * n_trials
        trial_data['tracker_v'] = [None] * n_trials
        trial_data['keypress'] = [None] * n_trials

        if savedata:
            trial_data['brain_state_a1'] = [None] * n_trials
            trial_data['derivatives_a1'] = [None] * n_trials
            trial_data['input_a1'] = [None] * n_trials
            trial_data['output_a1'] = [None] * n_trials
            trial_data['button_state_a1'] = [None] * n_trials

            trial_data['brain_state_a2'] = [None] * n_trials
            trial_data['derivatives_a2'] = [None] * n_trials
            trial_data['input_a2'] = [None] * n_trials
            trial_data['output_a2'] = [None] * n_trials
            trial_data['button_state_a2'] = [None] * n_trials

        n_trials = len(trials)
        for i in range(n_trials):
            # create target and tracker
            # target = Target(trials[i][0], self.step_size, 0)
            target = Target(trials[i][0], self.step_size, trials[i][2])

            if self.velocity_control == "buttons":
                tracker = Tracker(trials[i][1], self.step_size, self.condition)
            elif self.velocity_control == "direct":
                tracker = DirectTracker(None, self.step_size, self.condition)


            if lesion_point == "start":
                lesion_point = self.start_period
            elif lesion_point == "before_midturn":
                lesion_point = self.sim_length[i]/2
            elif lesion_point == "midturn":
                lesion_point = self.sim_length[i]/2 + self.start_period

            # set initial state in specified range
            agent1.brain.randomize_state(self.initial_state)
            agent1.initialize_buttons()

            agent2.brain.randomize_state(self.initial_state)
            agent2.initialize_buttons()

            trial_data['target_pos'][i] = np.zeros((self.sim_length[i] + self.start_period, 1))
            trial_data['tracker_pos'][i] = np.zeros((self.sim_length[i] + self.start_period, 1))
            trial_data['tracker_v'][i] = np.zeros((self.sim_length[i] + self.start_period, 1))
            trial_data['keypress'][i] = np.zeros((self.sim_length[i] + self.start_period, 2))

            if savedata:
                trial_data['brain_state_a1'][i] = np.zeros((self.sim_length[i] + self.start_period, agent1.brain.N))
                trial_data['derivatives_a1'][i] = np.zeros((self.sim_length[i] + self.start_period, agent1.brain.N))
                trial_data['input_a1'][i] = np.zeros((self.sim_length[i] + self.start_period, agent1.brain.N))
                trial_data['output_a1'][i] = np.zeros((self.sim_length[i] + self.start_period, 2))
                trial_data['button_state_a1'][i] = np.zeros((self.sim_length[i] + self.start_period, 2))

                trial_data['brain_state_a2'][i] = np.zeros((self.sim_length[i] + self.start_period, agent2.brain.N))
                trial_data['derivatives_a2'][i] = np.zeros((self.sim_length[i] + self.start_period, agent2.brain.N))
                trial_data['input_a2'][i] = np.zeros((self.sim_length[i] + self.start_period, agent2.brain.N))
                trial_data['output_a2'][i] = np.zeros((self.sim_length[i] + self.start_period, 2))
                trial_data['button_state_a2'][i] = np.zeros((self.sim_length[i] + self.start_period, 2))

            if self.start_period > 0:
                # don't move the target and don't allow tracker to move
                for j in range(self.start_period):
                    agent1.visual_input(tracker.position, target.position)
                    agent1.brain.euler_step()
                    agent2.visual_input(tracker.position, target.position)
                    agent2.brain.euler_step()

                    # activation, motor_activity = agent.motor_output()
                    # activation = agent.motor_output()
                    # tracker.accelerate(activation)

                    trial_data['target_pos'][i][j] = target.position
                    trial_data['tracker_pos'][i][j] = tracker.position
                    trial_data['tracker_v'][i][j] = tracker.velocity
                    # trial_data['keypress'][i][j] = activation

                    if savedata:
                        trial_data['brain_state_a1'][i][j] = agent1.brain.Y
                        trial_data['derivatives_a1'][i][j] = agent1.brain.dy_dt
                        trial_data['input_a1'][i][j] = agent1.brain.I
                        trial_data['brain_state_a2'][i][j] = agent2.brain.Y
                        trial_data['derivatives_a2'][i][j] = agent2.brain.dy_dt
                        trial_data['input_a2'][i][j] = agent2.brain.I

                        # trial_data['output'][i][j] = motor_activity
                        trial_data['button_state_a1'][i][j] = agent1.button_state
                        trial_data['button_state_a2'][i][j] = agent2.button_state

            for j in range(self.start_period, self.sim_length[i] + self.start_period):

                # 1) Target movement
                target.movement(self.width)
                # target.movement([self.width[0] + 10, self.width[1] - 10])

                # 2) Agent sees
                if (lesion_type == "visual_border" or lesion_type == "visual_target") and \
                                j > (lesion_point):
                    agent1.lesioned_visual_input(tracker.position, target.position, lesion_type)
                    agent2.lesioned_visual_input(tracker.position, target.position, lesion_type)
                else:
                    agent1.visual_input(tracker.position, target.position)
                    agent2.visual_input(tracker.position, target.position)

                # 3) Agents moves
                sound_output = tracker.movement(self.width)

                # 4) Agent hears
                if self.condition == 'sound':
                    if lesion_type == "auditory" and j > (lesion_point):
                        agent1.auditory_input([0, 0])
                        agent2.auditory_input([0, 0])
                    else:
                        agent1.auditory_input(sound_output)
                        agent2.auditory_input(sound_output)

                trial_data['target_pos'][i][j] = target.position
                trial_data['tracker_pos'][i][j] = tracker.position
                trial_data['tracker_v'][i][j] = tracker.velocity

                if savedata:
                    trial_data['brain_state_a1'][i][j] = agent1.brain.Y
                    trial_data['brain_state_a2'][i][j] = agent2.brain.Y
                    trial_data['derivatives_a1'][i][j] = agent1.brain.dy_dt
                    trial_data['derivatives_a2'][i][j] = agent2.brain.dy_dt

                # 5) Update agent's neural system
                agent1.brain.euler_step()
                agent2.brain.euler_step()

                # 6) Agent reacts
                activation1, neuron_output1 = agent1.motor_output()
                activation2, neuron_output2 = agent2.motor_output()
                activation_left = activation1[0]  # take only left activation from "left" agent
                activation_right = activation2[1]  # take only right activation from "right" agent

                activation = [activation_left, activation_right]
                tracker.accelerate(activation)
                # this will save -1 or 1 for button-controlling agents
                # but left and right velocities for direct velocity control agent
                trial_data['keypress'][i][j] = activation

                if savedata:
                    trial_data['input_a1'][i][j] = agent1.brain.I
                    trial_data['output_a1'][i][j] = neuron_output1
                    trial_data['button_state_a1'][i][j] = agent1.button_state
                    trial_data['input_a2'][i][j] = agent2.brain.I
                    trial_data['output_a2'][i][j] = neuron_output2
                    trial_data['button_state_a2'][i][j] = agent2.button_state

            # 6) Fitness tacking:
            fitness = 1 - (np.sum(np.abs(trial_data['target_pos'][i] - trial_data['tracker_pos'][i])) /
                           (2 * self.width[1] * (self.sim_length[i] + self.start_period)))
            # penalty for not moving in the trial not counting the delay period
            penalty = list(trial_data['tracker_v'][i][self.start_period:]).count(0) / (self.sim_length[i])
            # if penalty decreases the score below 0, set it to 0
            overall_fitness = np.clip(fitness - penalty, 0, 1)
            trial_data['fitness'].append(overall_fitness)

        return trial_data


class SimpleSimulation:
    """
    This is a class that implements a simple simulation version, in which the target appears at a random position
    and remains immobile for the duration of the trial. The tracker's task is to approach the target and stay close.
    """
    def __init__(self, step_size, evaluation_params):
        self.width = evaluation_params['screen_width']  # [-20, 20]
        self.step_size = step_size  # how fast things are happening in the simulation
        self.trials = self.create_trials(5)
        self.sim_length = 1000
        self.condition = evaluation_params['condition']  # is it a sound condition?
        # the period of time at the beginning of the trial in which the target stays still
        self.initial_state = evaluation_params['initial_state']
        self.velocity_control = evaluation_params['velocity_control']

    @staticmethod
    def create_trials(size):
        """
        Create a list of trials the environment will run.
        :return: 
        """
        left_positions = np.random.choice(np.arange(-20, 0), size, replace=False)
        right_positions = np.random.choice(np.arange(1, 21), size, replace=False)
        target_positions = np.concatenate((left_positions, right_positions))
        return target_positions

    def run_trials(self, agent, trials, savedata=False):
        """
        An evaluation function that accepts an agent and returns a real number representing
        the performance of that parameter vector on the task. Here the task is the Knoblich and Jordan task.

        :param agent: an agent with a CTRNN brain and particular anatomy
        :param trials: a list of trials to perform
        :param savedata: should the trial data be saved
        :return: fitness
        """

        n_trials = len(trials)
        trial_data = dict()
        trial_data['fitness'] = []
        trial_data['target_pos'] = [None] * n_trials
        trial_data['tracker_pos'] = [None] * n_trials
        trial_data['tracker_v'] = [None] * n_trials
        trial_data['keypress'] = [None] * n_trials

        if savedata:
            trial_data['brain_state'] = [None] * n_trials
            trial_data['derivatives'] = [None] * n_trials
            trial_data['input'] = [None] * n_trials
            trial_data['output'] = [None] * n_trials
            trial_data['button_state'] = [None] * n_trials

        for i in range(n_trials):
            # create target and tracker
            target = ImmobileTarget(trials[i])
            if self.velocity_control == "buttons":
                tracker = Tracker(1, self.step_size, self.condition)
            elif self.velocity_control == "direct":
                tracker = DirectTracker(None, self.step_size, self.condition)
            # set initial state in specified range
            agent.brain.randomize_state(self.initial_state)
            agent.initialize_buttons()

            trial_data['target_pos'][i] = np.zeros((self.sim_length, 1))
            trial_data['tracker_pos'][i] = np.zeros((self.sim_length, 1))
            trial_data['tracker_v'][i] = np.zeros((self.sim_length, 1))
            trial_data['keypress'][i] = np.zeros((self.sim_length, 2))

            if savedata:
                trial_data['brain_state'][i] = np.zeros((self.sim_length, agent.brain.N))
                trial_data['derivatives'][i] = np.zeros((self.sim_length, agent.brain.N))
                trial_data['input'][i] = np.zeros((self.sim_length, agent.brain.N))
                trial_data['output'][i] = np.zeros((self.sim_length, 2))
                trial_data['button_state'][i] = np.zeros((self.sim_length, 2))

            for j in range(self.sim_length):

                # 2) Agent sees
                agent.visual_input(tracker.position, target.position)

                # 3) Agents moves
                sound_output = tracker.movement(self.width)

                # 4) Agent hears
                if self.condition == 'sound':
                    agent.auditory_input(sound_output)

                trial_data['target_pos'][i][j] = target.position
                trial_data['tracker_pos'][i][j] = tracker.position
                trial_data['tracker_v'][i][j] = tracker.velocity

                if savedata:
                    trial_data['brain_state'][i][j] = agent.brain.Y
                    trial_data['derivatives'][i][j] = agent.brain.dy_dt

                # 5) Update agent's neural system
                agent.brain.euler_step()

                # 6) Agent reacts
                activation, neuron_output = agent.motor_output()
                # activation = agent.motor_output()
                tracker.accelerate(activation)
                # this will save -1 or 1 for button-controlling agents
                # but left and right velocities for direct velocity control agent
                trial_data['keypress'][i][j] = activation

                if savedata:
                    trial_data['input'][i][j] = agent.brain.I
                    trial_data['output'][i][j] = neuron_output
                    trial_data['button_state'][i][j] = agent.button_state

            # 6) Fitness tacking:
            fitness = 1 - (np.sum(np.abs(trial_data['target_pos'][i] - trial_data['tracker_pos'][i])) /
                           (2*self.width[1]*self.sim_length))
            # penalty for not moving in the trial not counting the delay period
            penalty = list(trial_data['tracker_v'][i]).count(0)/self.sim_length
            # if penalty decreases the score below 0, set it to 0
            overall_fitness = np.clip(fitness - penalty, 0, 1)
            trial_data['fitness'].append(overall_fitness)

        return trial_data

    def run_joint_trials(self, agent1, agent2, trials, savedata=False):
        """
        An evaluation function that accepts an agent and returns a real number representing
        the performance of that parameter vector on the task. Here the task is the Knoblich and Jordan task.

        :param agent: an agent with a CTRNN brain and particular anatomy
        :param trials: a list of trials to perform
        :param savedata: should the trial data be saved
        :return: fitness
        """

        n_trials = len(trials)
        trial_data = dict()
        trial_data['fitness'] = []
        trial_data['target_pos'] = [None] * n_trials
        trial_data['tracker_pos'] = [None] * n_trials
        trial_data['tracker_v'] = [None] * n_trials
        trial_data['keypress'] = [None] * n_trials

        if savedata:
            trial_data['brain_state_a1'] = [None] * n_trials
            trial_data['derivatives_a1'] = [None] * n_trials
            trial_data['input_a1'] = [None] * n_trials
            trial_data['output_a1'] = [None] * n_trials
            trial_data['button_state_a1'] = [None] * n_trials

            trial_data['brain_state_a2'] = [None] * n_trials
            trial_data['derivatives_a2'] = [None] * n_trials
            trial_data['input_a2'] = [None] * n_trials
            trial_data['output_a2'] = [None] * n_trials
            trial_data['button_state_a2'] = [None] * n_trials

        for i in range(n_trials):
            # create target and tracker
            target = ImmobileTarget(trials[i])
            if self.velocity_control == "buttons":
                tracker = Tracker(1, self.step_size, self.condition)
            elif self.velocity_control == "direct":
                tracker = DirectTracker(None, self.step_size, self.condition)
                tracker.position = self.initial_tracker_pos
            # set initial state in specified range
            agent1.brain.randomize_state(self.initial_state)
            agent1.initialize_buttons()

            agent2.brain.randomize_state(self.initial_state)
            agent2.initialize_buttons()

            trial_data['target_pos'][i] = np.zeros((self.sim_length, 1))
            trial_data['tracker_pos'][i] = np.zeros((self.sim_length, 1))
            trial_data['tracker_v'][i] = np.zeros((self.sim_length, 1))
            trial_data['keypress'][i] = np.zeros((self.sim_length, 2))

            if savedata:
                trial_data['brain_state_a1'][i] = np.zeros((self.sim_length, agent1.brain.N))
                trial_data['derivatives_a1'][i] = np.zeros((self.sim_length, agent1.brain.N))
                trial_data['input_a1'][i] = np.zeros((self.sim_length, agent1.brain.N))
                trial_data['output_a1'][i] = np.zeros((self.sim_length, agent1.brain.N))
                trial_data['button_state_a1'][i] = np.zeros((self.sim_length, 2))

                trial_data['brain_state_a2'][i] = np.zeros((self.sim_length, agent2.brain.N))
                trial_data['derivatives_a2'][i] = np.zeros((self.sim_length, agent2.brain.N))
                trial_data['input_a2'][i] = np.zeros((self.sim_length, agent2.brain.N))
                trial_data['output_a2'][i] = np.zeros((self.sim_length, agent2.brain.N))
                trial_data['button_state_a2'][i] = np.zeros((self.sim_length, 2))

            for j in range(self.sim_length):

                # 2) Agent sees
                agent1.visual_input(tracker.position, target.position)
                agent2.visual_input(tracker.position, target.position)

                # 3) Agents moves
                sound_output = tracker.movement(self.width)

                # 4) Agent hears
                if self.condition == 'sound':
                    agent1.auditory_input(sound_output)
                    agent2.auditory_input(sound_output)

                trial_data['target_pos'][i][j] = target.position
                trial_data['tracker_pos'][i][j] = tracker.position
                trial_data['tracker_v'][i][j] = tracker.velocity

                if savedata:
                    trial_data['brain_state_a1'][i][j] = agent1.brain.Y
                    trial_data['derivatives_a1'][i][j] = agent1.brain.dy_dt

                    trial_data['brain_state_a2'][i][j] = agent2.brain.Y
                    trial_data['derivatives_a2'][i][j] = agent2.brain.dy_dt

                # 5) Update agent's neural system
                agent1.brain.euler_step()
                agent2.brain.euler_step()

                # 6) Agent reacts
                activation1, neuron_output1 = agent1.motor_output()
                activation2, neuron_output2 = agent2.motor_output()
                activation_left = activation1[0]
                activation_right = activation2[1]
                activation = [activation_left, activation_right]
                tracker.accelerate(activation)
                # this will save -1 or 1 for button-controlling agents
                # but left and right velocities for direct velocity control agent
                trial_data['keypress'][i][j] = activation

                if savedata:

                    trial_data['input_a1'][i][j] = agent1.brain.I
                    trial_data['output_a1'][i][j] = expit(agent1.brain.Y + agent1.brain.Theta)
                    trial_data['button_state_a1'][i][j] = agent1.button_state

                    trial_data['input_a2'][i][j] = agent2.brain.I
                    trial_data['output_a2'][i][j] = expit(agent2.brain.Y + agent2.brain.Theta)
                    trial_data['button_state_a2'][i][j] = agent2.button_state

            # 6) Fitness tacking:
            fitness = 1 - (np.sum(np.abs(trial_data['target_pos'][i] - trial_data['tracker_pos'][i])) /
                           (2*self.width[1]*self.sim_length))
            # penalty for not moving in the trial not counting the delay period
            penalty = list(trial_data['tracker_v'][i]).count(0)/self.sim_length
            # if penalty decreases the score below 0, set it to 0
            overall_fitness = np.clip(fitness - penalty, 0, 1)
            trial_data['fitness'].append(overall_fitness)

        return trial_data


class SteadySimulation:
    """
    This is a class that implements a simple simulation version, in which the target appears at a random position
    and remains immobile for the duration of the trial. The tracker is also immobile at different positions.
    """
    def __init__(self, step_size, evaluation_params, tracker_start):
        self.width = evaluation_params['screen_width']  # [-20, 20]
        self.step_size = step_size  # how fast things are happening in the simulation
        self.trials = self.create_trials()
        self.sim_length = 3000
        self.condition = evaluation_params['condition']  # is it a sound condition?
        self.initial_state = evaluation_params['initial_state']
        self.velocity_control = evaluation_params['velocity_control']
        self.initial_tracker_pos = tracker_start

    @staticmethod
    def create_trials():
        """
        Create a list of trials the environment will run.
        :return:
        """
        target_positions = list(range(-20, 21))
        return target_positions


    def run_joint_trials(self, agent1, agent2, trials, savedata=False):
        n_trials = len(trials)
        trial_data = dict()
        trial_data['target_pos'] = [None] * n_trials
        trial_data['tracker_pos'] = [None] * n_trials
        trial_data['tracker_v'] = [None] * n_trials
        trial_data['keypress'] = [None] * n_trials

        if savedata:
            trial_data['brain_state_a1'] = [None] * n_trials
            trial_data['derivatives_a1'] = [None] * n_trials
            trial_data['input_a1'] = [None] * n_trials
            trial_data['output_a1'] = [None] * n_trials
            trial_data['button_state_a1'] = [None] * n_trials

            trial_data['brain_state_a2'] = [None] * n_trials
            trial_data['derivatives_a2'] = [None] * n_trials
            trial_data['input_a2'] = [None] * n_trials
            trial_data['output_a2'] = [None] * n_trials
            trial_data['button_state_a2'] = [None] * n_trials

        for i in range(n_trials):
            # create target and tracker
            target = ImmobileTarget(trials[i])
            if self.velocity_control == "buttons":
                tracker = Tracker(1, self.step_size, self.condition)
            elif self.velocity_control == "direct":
                tracker = ImmobileTracker(None, self.step_size, self.condition)
                tracker.position = self.initial_tracker_pos
            # set initial state in specified range
            agent1.brain.randomize_state(self.initial_state)
            agent1.initialize_buttons()

            agent2.brain.randomize_state(self.initial_state)
            agent2.initialize_buttons()

            trial_data['target_pos'][i] = np.zeros((self.sim_length, 1))
            trial_data['tracker_pos'][i] = np.zeros((self.sim_length, 1))
            trial_data['tracker_v'][i] = np.zeros((self.sim_length, 1))
            trial_data['keypress'][i] = np.zeros((self.sim_length, 2))

            if savedata:
                trial_data['brain_state_a1'][i] = np.zeros((self.sim_length, agent1.brain.N))
                trial_data['derivatives_a1'][i] = np.zeros((self.sim_length, agent1.brain.N))
                trial_data['input_a1'][i] = np.zeros((self.sim_length, agent1.brain.N))
                trial_data['output_a1'][i] = np.zeros((self.sim_length, agent1.brain.N))
                trial_data['button_state_a1'][i] = np.zeros((self.sim_length, 2))

                trial_data['brain_state_a2'][i] = np.zeros((self.sim_length, agent2.brain.N))
                trial_data['derivatives_a2'][i] = np.zeros((self.sim_length, agent2.brain.N))
                trial_data['input_a2'][i] = np.zeros((self.sim_length, agent2.brain.N))
                trial_data['output_a2'][i] = np.zeros((self.sim_length, agent2.brain.N))
                trial_data['button_state_a2'][i] = np.zeros((self.sim_length, 2))

            for j in range(self.sim_length):

                # 2) Agent sees
                agent1.visual_input(tracker.position, target.position)
                agent2.visual_input(tracker.position, target.position)

                # 3) Agents moves
                sound_output = tracker.movement(self.width)

                # 4) Agent hears
                if self.condition == 'sound':
                    agent1.auditory_input(sound_output)
                    agent2.auditory_input(sound_output)

                trial_data['target_pos'][i][j] = target.position
                trial_data['tracker_pos'][i][j] = tracker.position
                trial_data['tracker_v'][i][j] = tracker.velocity

                if savedata:
                    trial_data['brain_state_a1'][i][j] = agent1.brain.Y
                    trial_data['derivatives_a1'][i][j] = agent1.brain.dy_dt

                    trial_data['brain_state_a2'][i][j] = agent2.brain.Y
                    trial_data['derivatives_a2'][i][j] = agent2.brain.dy_dt

                # 5) Update agent's neural system
                agent1.brain.euler_step()
                agent2.brain.euler_step()

                # 6) Agent reacts
                activation1, neuron_output1 = agent1.motor_output()
                activation2, neuron_output2 = agent2.motor_output()
                activation_left = activation1[0]
                activation_right = activation2[1]
                activation = [activation_left, activation_right]
                tracker.accelerate(activation)
                # this will save -1 or 1 for button-controlling agents
                # but left and right velocities for direct velocity control agent
                trial_data['keypress'][i][j] = activation

                if savedata:

                    trial_data['input_a1'][i][j] = agent1.brain.I
                    trial_data['output_a1'][i][j] = expit(agent1.brain.Y + agent1.brain.Theta)
                    trial_data['button_state_a1'][i][j] = agent1.button_state

                    trial_data['input_a2'][i][j] = agent2.brain.I
                    trial_data['output_a2'][i][j] = expit(agent2.brain.Y + agent2.brain.Theta)
                    trial_data['button_state_a2'][i][j] = agent2.button_state

        return trial_data


class ImmobileTarget:
    def __init__(self, position):
        self.position = position


class Target:
    """
    Target moves with constant velocity and starts from the middle.
    Target velocity varies across the trials and can be set to negative or positive values, in which case
    the target's initial movement direction is left or right respectively.
    """

    def __init__(self, velocity, step_size, start_pos):
        self.position = start_pos
        self.velocity = velocity
        self.step_size = step_size

    def reverse_direction(self, border_range, future_pos):
        """
        Reverse target direction if going beyond the border.
        :param border_range: border positions of the environment
        :param future_pos: predicted position at the next time step
        :return: 
        """
        if ((self.velocity > 0) and (future_pos > border_range[1])) or \
                ((self.velocity < 0) and (future_pos < border_range[0])):
            self.velocity *= -1

    def movement(self, border_range):
        future_pos = self.position + self.velocity * self.step_size
        self.reverse_direction(border_range, future_pos)
        self.position += self.velocity * self.step_size


class Tracker:
    """
    Tracker moves as a result of its set velocity and can accelerate based on agent button clicks.
    It starts in the middle of the screen and with initial 0 velocity.
    """

    def __init__(self, impact, step_size, condition):
        self.position = 0
        self.velocity = 0
        self.impact = impact  # how much acceleration is added by button click
        self.step_size = step_size
        # Timer for the emitted sound-feedback
        self.condition = condition  # is it a sound condition?
        self.timer_sound_l = 0
        self.timer_sound_r = 0

    def movement(self, border_range):
        """ Update self.position and self.timer(sound) """

        self.position += self.velocity * self.step_size

        # Tacker does not continue moving, when at the edges of the environment.
        if self.position < border_range[0]:
            self.position = border_range[0]
            self.velocity = 0  # added by GK
        if self.position > border_range[1]:
            self.position = border_range[1]
            self.velocity = 0  # added by GK

        sound_output = [0, 0]

        if self.timer_sound_l > 0:
            self.timer_sound_l -= self.step_size
            sound_output[0] = 1   # auditory feedback from the left button click

        if self.timer_sound_r > 0:
            self.timer_sound_r -= self.step_size
            sound_output[1] = 1  # auditory feedback from the right button click

        return sound_output

    def accelerate(self, inputs):
        """
        Accelerates the tracker to the left or to the right
        Impact of the keypress (how much the velocity is changed) is controlled
        by the tracker's impact attribute.
        :param inputs: an array of size two with values of -1 or +1 (left or right)
        :return: update self.velocity
        """
        acceleration = np.dot(np.array([self.impact, self.impact]).T, inputs)
        self.velocity += acceleration

        if self.condition == "sound":
            self.set_timer(inputs)

    def set_timer(self, left_or_right):
        """ Emit tone of 100-ms duration """
        if left_or_right[0] == -1:  # left
            self.timer_sound_l = 0.1
        if left_or_right[1] == 1:   # right
            self.timer_sound_r = 0.1


class DirectTracker(Tracker):
    """
    DirectTracker moves as a result of its set velocity, which can change based on agent's output
    to left and right motors.
    It starts in the middle of the screen and with initial 0 velocity.
    """

    def __init__(self, impact, step_size, condition):
        Tracker.__init__(self, impact, step_size, condition)

    def accelerate(self, inputs):
        """
        Sets the tracker velocity depending on the activation of the agent's motor neurons.
        The overall velocity is a difference between left and right velocities.
        Whenever the right velocity is greater than left, the tracker moves right and vice versa.
        :param inputs: an array of size two with activation values
        :return: update self.velocity
        """
        # new_velocity = inputs[1] - inputs[0]
        new_velocity = inputs[0] + inputs[1]
        # velocity_change = new_velocity - self.velocity

        if self.condition == "sound":
            # self.set_timer(velocity_change)
            self.set_timer(inputs)

        self.velocity = new_velocity

    # def set_timer(self, velocity_change):
    #     """ Emit tone of 100-ms duration """
    #     if velocity_change > 0:
    #         self.timer_sound_r = 0.1
    #     elif velocity_change < 0:
    #         self.timer_sound_l = 0.1

    # def set_timer(self, inputs):
    #     """ Emit tone of 100-ms duration """
    #     if inputs[1] > inputs[0]:  # are we trying to move right?
    #         self.timer_sound_r = 0.1
    #     elif inputs[1] < inputs[0]:
    #         self.timer_sound_l = 0.1
    #     else:
    #         pass

    def set_timer(self, inputs):
        """ Emit tones proportional to the motor activation """
        self.timer_sound_l = inputs[0] * 0.1
        self.timer_sound_r = inputs[1] * 0.1

    def movement(self, border_range):
        """ Update self.position and self.timer(sound) """

        self.position += self.velocity * self.step_size

        # Tacker does not continue moving, when at the edges of the environment.
        if self.position < border_range[0]:
            self.position = border_range[0]
            self.velocity = 0  # added by GK
        if self.position > border_range[1]:
            self.position = border_range[1]
            self.velocity = 0  # added by GK

        sound_output = [self.timer_sound_l, self.timer_sound_r]

        return sound_output


class ImmobileTracker(Tracker):
    """
    ImmobileTracker does not move but continues to receive sound input.
    It starts in the middle of the screen and with initial 0 velocity.
    """

    def __init__(self, impact, step_size, condition):
        Tracker.__init__(self, impact, step_size, condition)

    def accelerate(self, inputs):
        """
        Sets the tracker velocity depending on the activation of the agent's motor neurons.
        The overall velocity is a difference between left and right velocities.
        Whenever the right velocity is greater than left, the tracker moves right and vice versa.
        :param inputs: an array of size two with activation values
        :return: update self.velocity
        """
        new_velocity = inputs[0] + inputs[1]
        if self.condition == "sound":
            # self.set_timer(velocity_change)
            self.set_timer(inputs)
        self.velocity = new_velocity

    def set_timer(self, inputs):
        """ Emit tones proportional to the motor activation """
        self.timer_sound_l = inputs[0] * 0.1
        self.timer_sound_r = inputs[1] * 0.1

    def movement(self, border_range):
        """ Update self.position and self.timer(sound) """

        self.position = self.position

        # Tacker does not continue moving, when at the edges of the environment.
        if self.position < border_range[0]:
            self.position = border_range[0]
        if self.position > border_range[1]:
            self.position = border_range[1]

        sound_output = [self.timer_sound_l, self.timer_sound_r]

        return sound_output


# def setup_trials(step_size, evaluation_params):
#     width = evaluation_params['screen_width']
#     step_size = evaluation_params['step_size']
#     trials = create_trials(evaluation_params['velocities'], evaluation_params['impacts'],
#                            evaluation_params['tg_start_range'], evaluation_params['tg_start_variants'])
#     distance = (width[1]-width[0]) * evaluation_params['n_turns']  # total distance travelled by target
#     # simulation length depends on target velocity
#     sim_length = [int(distance/abs(trial[0])/step_size) for trial in trials]
#     evaluation_params["sim_length"] = sim_length
#     evaluation_params["trials"] = trials
#     return evaluation_params
#
#
# def create_trials(velocities, impacts, start_range, size):
#     """
#     Create a list of trials the environment will run.
#     :return:
#     """
#     if start_range[1]-start_range[0] == 0:
#         trials = [(x, y, start_range[0]) for x in velocities for y in impacts]
#     else:
#         left_positions = np.random.choice(np.arange(start_range[0], 1), size, replace=False)
#         right_positions = np.random.choice(np.arange(0, start_range[1] + 1), size, replace=False)
#         target_positions = np.concatenate((left_positions, right_positions))
#         trials = [(x, y, z) for x in velocities for y in impacts for z in target_positions]
#     return trials
#
#
# def run_joint_trials(agent1, agent2, params, savedata=False):
#     """
#     An evaluation function that accepts two agents and returns a real number representing
#     the performance of these agents on the task.
#     Here the task is the Knoblich and Jordan task, the joint action version.
#
#     :param agent1: an agent with a CTRNN brain and particular anatomy, controls only left button
#     :param agent2: an agent with a CTRNN brain and particular anatomy, controls only right button
#     :param trials: a list of trials to perform
#     :param savedata: should all the trial data be saved
#     :return: fitness
#     """
#
#
#     step_size = params["step_size"]
#     condition = params["condition"]
#     velocity_control = params["velocity_control"]
#     initial_state = params["initial_state"]
#     sim_length = params["sim_length"]
#     start_period = params["start_period"]
#     width = params["screen_width"]
#     trials = params["trials"]
#
#     cdef int i
#     cdef int n_trials = len(trials)
#
#
#     trial_data = dict()
#     trial_data['fitness'] = []
#     trial_data['target_pos'] = [None] * n_trials
#     trial_data['tracker_pos'] = [None] * n_trials
#     trial_data['tracker_v'] = [None] * n_trials
#     trial_data['keypress'] = [None] * n_trials
#
#     if savedata:
#         trial_data['brain_state_a1'] = [None] * n_trials
#         trial_data['input_a1'] = [None] * n_trials
#         trial_data['brain_state_a2'] = [None] * n_trials
#         trial_data['input_a2'] = [None] * n_trials
#         trial_data['output'] = [None] * n_trials
#         trial_data['button_state_a1'] = [None] * n_trials
#         trial_data['button_state_a2'] = [None] * n_trials
#
#     for i in range(n_trials):
#         # create target and tracker
#         # target = Target(trials[i][0], self.step_size, 0)
#         target = Target(trials[i][0], step_size, trials[i][2])
#
#         if velocity_control == "buttons":
#             tracker = Tracker(trials[i][1], step_size, condition)
#         elif velocity_control == "direct":
#             tracker = DirectTracker(None, step_size, condition)
#
#         # set initial state in specified range
#         agent1.brain.randomize_state(initial_state)
#         agent1.initialize_buttons()
#
#         agent2.brain.randomize_state(initial_state)
#         agent2.initialize_buttons()
#
#         trial_data['target_pos'][i] = np.zeros((sim_length[i] + start_period, 1))
#         trial_data['tracker_pos'][i] = np.zeros((sim_length[i] + start_period, 1))
#         trial_data['tracker_v'][i] = np.zeros((sim_length[i] + start_period, 1))
#         trial_data['keypress'][i] = np.zeros((sim_length[i] + start_period, 2))
#
#         if savedata:
#             trial_data['brain_state_a1'][i] = np.zeros((sim_length[i] + start_period, agent1.brain.N))
#             trial_data['input_a1'][i] = np.zeros((sim_length[i] + start_period, agent1.brain.N))
#             trial_data['brain_state_a2'][i] = np.zeros((sim_length[i] + start_period, agent2.brain.N))
#             trial_data['input_a2'][i] = np.zeros((sim_length[i] + start_period, agent2.brain.N))
#             trial_data['output'][i] = np.zeros((sim_length[i] + start_period, 2))
#             trial_data['button_state_a1'][i] = np.zeros((sim_length[i] + start_period, 2))
#             trial_data['button_state_a2'][i] = np.zeros((sim_length[i] + start_period, 2))
#
#         if start_period > 0:
#             # don't move the target and don't allow tracker to move
#             for j in range(start_period):
#                 agent1.visual_input(tracker.position, target.position)
#                 agent1.brain.euler_step()
#                 agent2.visual_input(tracker.position, target.position)
#                 agent2.brain.euler_step()
#
#                 trial_data['target_pos'][i][j] = target.position
#                 trial_data['tracker_pos'][i][j] = tracker.position
#                 trial_data['tracker_v'][i][j] = tracker.velocity
#                 # trial_data['keypress'][i][j] = activation
#
#                 if savedata:
#                     trial_data['brain_state_a1'][i][j] = agent1.brain.Y
#                     trial_data['input_a1'][i][j] = agent1.brain.I
#                     trial_data['brain_state_a2'][i][j] = agent2.brain.Y
#                     trial_data['input_a2'][i][j] = agent2.brain.I
#                     # trial_data['output'][i][j] = motor_activity
#                     trial_data['button_state_a1'][i][j] = agent1.button_state
#                     trial_data['button_state_a2'][i][j] = agent2.button_state
#
#         for j in range(start_period, sim_length[i] + start_period):
#
#             # 1) Target movement
#             target.movement(width)
#             # target.movement([self.width[0] + 10, self.width[1] - 10])
#
#             # 2) Agent sees
#             agent1.visual_input(tracker.position, target.position)
#             agent2.visual_input(tracker.position, target.position)
#
#             # 3) Agents moves
#             sound_output = tracker.movement(width)
#
#             # 4) Agent hears
#             if condition == 'sound':
#                 agent1.auditory_input(sound_output)
#                 agent2.auditory_input(sound_output)
#
#             trial_data['target_pos'][i][j] = target.position
#             trial_data['tracker_pos'][i][j] = tracker.position
#             trial_data['tracker_v'][i][j] = tracker.velocity
#
#             if savedata:
#                 trial_data['brain_state_a1'][i][j] = agent1.brain.Y
#                 trial_data['brain_state_a2'][i][j] = agent2.brain.Y
#
#             # 5) Update agent's neural system
#             agent1.brain.euler_step()
#             agent2.brain.euler_step()
#
#             # 6) Agent reacts
#             activation_left = agent1.motor_output()[0]  # take only left activation from "left" agent
#             activation_right = agent2.motor_output()[1]  # take only right activation from "right" agent
#
#             activation = [activation_left, activation_right]
#             tracker.accelerate(activation)
#             trial_data['keypress'][i][j] = activation
#
#             if savedata:
#                 trial_data['input_a1'][i][j] = agent1.brain.I
#                 # trial_data['output'][i][j] = motor_activity
#                 trial_data['button_state_a1'][i][j] = agent1.button_state
#                 trial_data['input_a2'][i][j] = agent2.brain.I
#                 # trial_data['output'][i][j] = motor_activity
#                 trial_data['button_state_a2'][i][j] = agent2.button_state
#
#         # 6) Fitness tacking:
#         fitness = 1 - (np.sum(np.abs(trial_data['target_pos'][i] - trial_data['tracker_pos'][i])) /
#                        (2 * width[1] * (sim_length[i] + start_period)))
#         # penalty for not moving in the trial not counting the delay period
#         penalty = list(trial_data['tracker_v'][i][start_period:]).count(0) / (sim_length[i])
#         # if penalty decreases the score below 0, set it to 0
#         overall_fitness = np.clip(fitness - penalty, 0, 1)
#         trial_data['fitness'].append(overall_fitness)
#
#     return trial_data
