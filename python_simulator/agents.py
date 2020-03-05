import numpy as np
import random
import string
from scipy.special import expit


class Agent:
    """
    This is a class that implements agents in the simulation. Agents' brains are CTRNN, but they also have
    a particular anatomy and a connection to external input and output.
    The anatomy is set up by the parameters defined in the config.json file: they specify the number of particular
    sensors and effectors the agent has and the number of connections each sensor has to other neurons, as well
    as the weight and gene ranges.
    """
    def __init__(self, network, agent_parameters):

        self.brain = network
        self.r_range = agent_parameters['r_range']  # receptor gain range
        self.e_range = agent_parameters['e_range']  # effector gain range

        self.VW = np.random.uniform(self.r_range[0], self.r_range[1],
                                    (agent_parameters['n_visual_sensors'] * agent_parameters['n_visual_connections']))
        self.AW = np.random.uniform(self.r_range[0], self.r_range[1],
                                    (agent_parameters['n_audio_sensors'] * agent_parameters['n_audio_connections']))
        self.MW = np.random.uniform(self.e_range[0], self.e_range[1],
                                    (agent_parameters['n_effectors'] * agent_parameters['n_effector_connections']))
        self.gene_range = agent_parameters['gene_range']
        self.genotype = self.make_genotype_from_params()
        self.fitness = 0.0
        self.n_io = len(self.VW) + len(self.AW) + len(self.MW)  # how many input-output weights

        self.timer_motor_l = 0.0
        self.timer_motor_r = 0.0

        # calculate crossover points
        self.n_evp = len(agent_parameters['evolvable_params'])  # how many parameters in addition to weights are evolved
        crossover_points = [i * (self.n_evp + self.brain.N) for i in range(1, self.brain.N + 1)]
        crossover_points.extend([crossover_points[-1] + len(self.VW),
                                 crossover_points[-1] + len(self.VW) + len(self.AW)])
        self.crossover_points = crossover_points
        self.button_state = [False, False]  # both buttons off in the beginning
        self.name = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))

    def __eq__(self, other):
        # two agents are the same if they have the same genotype
        if np.all(self.genotype == other.genotype):
            return True

    def initialize_buttons(self):
        self.button_state = [False, False]

    def make_genotype_from_params(self):
        """
        Combine all parameters and reshape into a single vector
        :return: [Tau_n1, G_n1, Theta_n1, W_n1..., all visual w, all auditory w, all motor w]
        """
        # return [self.Tau, self.G, self.Theta, self.W]
        tau = linmap(self.brain.Tau, self.brain.tau_range, self.gene_range)
        # skip G in evolution
        # g = self.linmap(self.brain.G, self.brain.g_range, [0, 1])
        theta = linmap(self.brain.Theta, self.brain.theta_range, self.gene_range)
        w = linmap(self.brain.W.T, self.brain.w_range, self.gene_range)
        vw = linmap(self.VW, self.r_range, self.gene_range)
        aw = linmap(self.AW, self.r_range, self.gene_range)
        mw = linmap(self.MW, self.e_range, self.gene_range)
        # this was also an option in some old code
        #  mw = np.concatenate([linmap(self.MW[:2], self.left_range, self.gene_range),
        #                       linmap(self.MW[2:], self.right_range, self.gene_range)])

        stacked = np.vstack((tau, theta, w))
        flattened = stacked.reshape(stacked.size, order='F')
        genotype = np.hstack((flattened, vw, aw, mw))
        return genotype

    def make_params_from_genotype(self, genotype):
        """
        Take a genotype vector and set all agent parameters.
        :param genotype: vector that represents agent parameters in the unified gene range.
        :return:
        """
        genorest, vw, aw, mw = np.hsplit(genotype, self.crossover_points[-3:])
        self.VW = linmap(vw, self.gene_range, self.r_range)
        self.AW = linmap(aw, self.gene_range, self.r_range)
        self.MW = linmap(mw, self.gene_range, self.e_range)
        # self.MW = np.concatenate([linmap(mw[:2], self.gene_range, self.left_range),
        #                           linmap(mw[2:], self.gene_range, self.right_range)])

        unflattened = genorest.reshape(self.n_evp+self.brain.N, self.brain.N, order='F')
        tau, theta, w = (np.squeeze(a) for a in np.vsplit(unflattened, [1, 2]))
        self.brain.Tau = linmap(tau, self.gene_range, self.brain.tau_range)
        self.brain.Theta = linmap(theta, self.gene_range, self.brain.theta_range)
        self.brain.W = linmap(w, self.gene_range, self.brain.w_range).transpose()

    def visual_input(self, position_tracker, position_target):
        """
        The visual input to the agent
        :param position_tracker: absolute position of the tracker
        :param position_target: absolute position of the target
        :return:
        """
        # add noise to the visual input
        position_tracker = add_noise(position_tracker)
        position_target = add_noise(position_target)

        self.brain.I[1] = self.VW[0] * position_target  # to n2
        self.brain.I[2] = self.VW[1] * position_tracker  # to n3
        self.brain.I[0] = self.VW[2] * position_target + self.VW[3] * position_tracker  # to n1

    def auditory_input(self, sound_input):
        """
        The auditory input to the agent
        :param sound_input: Tone(s) induced by left and/or right click
        """
        left_click, right_click = sound_input[0], sound_input[1]

        self.brain.I[3] = self.AW[0] * left_click  # to n4
        self.brain.I[5] = self.AW[1] * right_click  # to n6
        self.brain.I[4] = self.AW[2] * left_click + self.AW[3] * right_click  # to n5

    def motor_output(self):
        """
        The motor output of the agent
        :return: output
        """
        # Set activation threshold
        activation = [0, 0]  # Initial activation is zero
        # threshold = 0  # Threshold for output
        threshold = 0.5  # Threshold for output

        # consider adding noise to output before multiplying by motor gains,
        # drawn from a Gaussian distribution with (mu=0, var=0.05)
        o7 = self.brain.Y[6] + self.brain.Theta[6]  # output of n7
        o8 = self.brain.Y[7] + self.brain.Theta[7]  # output of n8

        activation_left = expit(o7 * self.MW[0] + o8 * self.MW[2])
        activation_right = expit(o7 * self.MW[1] + o8 * self.MW[3])

        # Update timer:
        if self.timer_motor_l > 0:
            self.timer_motor_l -= self.brain.step_size
        if self.timer_motor_r > 0:
            self.timer_motor_r -= self.brain.step_size

        # We set timer to 0.5. That means we have max. 2 clicks per time-unit
        if activation_left > threshold:
            if self.timer_motor_l <= 0:
                self.timer_motor_l = 0.5  # reset the timer
                activation[0] = -1  # set left activation to -1 to influence velocity to the left

        if activation_right > threshold:
            if self.timer_motor_r <= 0:
                self.timer_motor_r = 0.5
                activation[1] = 1  # set right to one to influence velocity to the right

        return activation


def linmap(vin, rin, rout):
    """
    Map a vector between 2 ranges.
    :param vin: input vector to be mapped
    :param rin: range of vin to map from
    :param rout: range to map to
    :return: mapped output vector
    :rtype np.ndarray
    """
    a = rin[0]
    b = rin[1]
    c = rout[0]
    d = rout[1]
    return ((c + d) + (d - c) * ((2 * vin - (a + b)) / (b - a))) / 2


def add_noise(state):
    magnitude = np.random.normal(0, 0.05)
    return state + magnitude


class EmbodiedAgentV1(Agent):
    """
    This is a version of the agent that is more embodied.
    The agent receives positions of target and environment borders in terms of their distance to the position
    of the tracker (which implements the agent's "body"). The agent also has a different anatomy with respect to
    the non-embodied agent, see network2.pdf for a picture.
    """
    def __init__(self, network, agent_parameters, screen_width):
        # change visual input: 3 distance sensors for border_left, border_right, target
        # each sensor connected with one connection to 3 different neurons (1, 2, 3)
        agent_parameters["n_visual_sensors"] = 3
        agent_parameters["n_visual_connections"] = 1
        agent_parameters["n_audio_sensors"] = 2
        agent_parameters["n_audio_connections"] = 1
        agent_parameters["n_effector_connections"] = 2

        Agent.__init__(self, network, agent_parameters)
        self.screen_width = screen_width

    def visual_input(self, position_tracker, position_target):
        """
        The visual input to the agent
        :param position_tracker: absolute position of the tracker
        :param position_target: absolute position of the target
        :return:
        """
        # distance to borders is absolute
        dleft_border = add_noise(abs(self.screen_width[0] - position_tracker))
        dright_border = add_noise(abs(self.screen_width[1] - position_tracker))
        # distance to target is not: neg when target is on the left, pos when on the right
        dtarget = add_noise(position_target - position_tracker)

        self.brain.I[0] = self.VW[0] * dtarget  # to n1
        self.brain.I[1] = self.VW[1] * dleft_border  # to n2
        self.brain.I[2] = self.VW[2] * dright_border  # to n3

    def auditory_input(self, sound_input):
        """
        The auditory input to the agent
        :param sound_input: Tone(s) induced by left and/or right click
        """
        left_click, right_click = sound_input[0], sound_input[1]

        self.brain.I[3] = self.AW[0] * left_click  # to n4
        self.brain.I[5] = self.AW[1] * right_click  # to n6

    def motor_output(self):
        """
        The motor output of the agent
        :return: output
        """
        # Set activation threshold
        activation = [0, 0]  # Initial activation is zero
        # threshold = 0  # Threshold for output
        threshold = 0.5  # Threshold for output

        # consider adding noise to output before multiplying by motor gains,
        # drawn from a Gaussian distribution with (mu=0, var=0.05)
        o7 = self.brain.Y[6] + self.brain.Theta[6]  # output of n7
        o8 = self.brain.Y[7] + self.brain.Theta[7]  # output of n8

        # activation_left = self.brain.expit(o7 * self.MW[0])
        # activation_right = self.brain.expit(o8 * self.MW[1])
        activation_left = expit(o7 * self.MW[0] + o8 * self.MW[2])
        activation_right = expit(o7 * self.MW[1] + o8 * self.MW[3])

        # Update timer:
        if self.timer_motor_l > 0:
            self.timer_motor_l -= self.brain.step_size
        if self.timer_motor_r > 0:
            self.timer_motor_r -= self.brain.step_size

        # We set timer to 0.5. That means we have max. 2 clicks per time-unit
        # Version by GK
        if activation_left >= threshold:
            if activation_right >= threshold:  # both over threshold: choose one in proportion to relative activation
                if random.random() <= activation_left / (activation_left + activation_right):
                    if self.timer_motor_l <= 0:  # left wins
                        self.timer_motor_l = 0.5
                        activation[0] = -1
                elif self.timer_motor_r <= 0:  # right wins
                    self.timer_motor_r = 0.5
                    activation[1] = 1
            elif self.timer_motor_l <= 0:  # only left is over threshold
                self.timer_motor_l = 0.5
                activation[0] = -1
        elif activation_right >= threshold:
            if self.timer_motor_r <= 0:
                self.timer_motor_r = 0.5
                activation[1] = 1

        return activation


class EmbodiedAgentV2(Agent):
    """
    This is a version of the agent that is more embodied.
    The agent receives positions of target and environment borders in terms of their distance to the position
    of the tracker (which implements the agent's "body"). Compared to V1 its perception is in terms of separate
    distance to the target on the left or right and inverted visual stimulation (the smaller the distance,
    the larger the stimulation; linearly scaled to be max 10).
    See network3.pdf for a picture.
    """
    def __init__(self, network, agent_parameters, screen_width):
        # change visual input: 4 absolute distance sensors for border_left, border_right, target_left, target_right
        # each sensor connected with 1 connection to 4 different neurons (1, 2, 3, 4)
        # each auditory sensor with 1 connection to 2 different neurons (5, 6)
        # each motor with 1 connection to 2 different neurons (7, 8)
        agent_parameters["n_visual_sensors"] = 4
        agent_parameters["n_visual_connections"] = 1
        agent_parameters["n_audio_sensors"] = 2
        agent_parameters["n_audio_connections"] = 1
        agent_parameters["n_effector_connections"] = 2

        Agent.__init__(self, network, agent_parameters)
        self.screen_width = screen_width
        self.max_dist = self.screen_width[1] - self.screen_width[0]
        self.visual_scale = self.max_dist / agent_parameters["max_visual_activation"]

    def visual_input(self, position_tracker, position_target):
        """
        The visual input to the agent
        :param position_tracker: absolute position of the tracker
        :param position_target: absolute position of the target
        :return:
        """
        dleft_border = add_noise((self.max_dist-abs(self.screen_width[0]-position_tracker))/self.visual_scale)
        dright_border = add_noise((self.max_dist-abs(self.screen_width[1]-position_tracker))/self.visual_scale)

        if position_target > position_tracker:
            # target is to the right of the tracker
            dleft_target = 0
            dright_target = add_noise((self.max_dist-abs(position_target-position_tracker))/self.visual_scale)
        elif position_target < position_tracker:
            # target is to the left of the tracker
            dleft_target = add_noise((self.max_dist-abs(position_target-position_tracker))/self.visual_scale)
            dright_target = 0
        else:
            # if tracker is on top of the target, both eyes are activated to the maximum
            dleft_target = add_noise((self.max_dist-abs(position_target-position_tracker))/self.visual_scale)
            dright_target = add_noise((self.max_dist-abs(position_target-position_tracker))/self.visual_scale)

        self.brain.I[0] = self.VW[0] * dleft_border  # to n1
        self.brain.I[1] = self.VW[1] * dright_border  # to n2
        self.brain.I[2] = self.VW[2] * dleft_target  # to n3
        self.brain.I[3] = self.VW[3] * dright_target  # to n4

    def auditory_input(self, sound_input):
        """
        The auditory input to the agent
        :param sound_input: Tone(s) induced by left and/or right click
        """
        left_click, right_click = sound_input[0], sound_input[1]

        self.brain.I[4] = self.AW[0] * left_click  # to n5
        self.brain.I[5] = self.AW[1] * right_click  # to n6

    def motor_output(self):
        """
        The motor output of the agent
        One neuron controls leftward, the other rightward velocity. Each velocity is calculated by multiplying
        the activation in the range of [0, 1] by output gain.
        :return: output
        """
        # Set activation threshold
        activation = [0, 0]  # Initial activation is zero
        # threshold = 0  # Threshold for output
        threshold = 0.5  # Threshold for output

        # consider adding noise to output before multiplying by motor gains,
        # drawn from a Gaussian distribution with (mu=0, var=0.05)
        o7 = self.brain.Y[6] + self.brain.Theta[6]  # output of n7
        o8 = self.brain.Y[7] + self.brain.Theta[7]  # output of n8

        # activation_left = self.brain.expit(o7 * self.MW[0])
        # activation_right = self.brain.expit(o8 * self.MW[1])
        activation_left = expit(o7 * self.MW[0] + o8 * self.MW[2])
        activation_right = expit(o7 * self.MW[1] + o8 * self.MW[3])

        # Update timer:
        if self.timer_motor_l > 0:
            self.timer_motor_l -= self.brain.step_size
        if self.timer_motor_r > 0:
            self.timer_motor_r -= self.brain.step_size

        # We set timer to 0.5. That means we have max. 2 clicks per time-unit
        # Version by GK
        if activation_left >= threshold:
            if activation_right >= threshold:  # both over threshold: choose one in proportion to relative activation
                if random.random() <= activation_left / (activation_left + activation_right):
                    if self.timer_motor_l <= 0:  # left wins
                        self.timer_motor_l = 0.5
                        activation[0] = -1
                elif self.timer_motor_r <= 0:  # right wins
                    self.timer_motor_r = 0.5
                    activation[1] = 1
            elif self.timer_motor_l <= 0:  # only left is over threshold
                self.timer_motor_l = 0.5
                activation[0] = -1
        elif activation_right >= threshold:
            if self.timer_motor_r <= 0:
                self.timer_motor_r = 0.5
                activation[1] = 1

        return activation


class ButtonOnOffAgent(EmbodiedAgentV2):
    """
    This is a version of the embodied agent (v2) with a modified button pressing mechanism.
    """
    def __init__(self, network, agent_parameters, screen_width):
        EmbodiedAgentV2.__init__(self, network, agent_parameters, screen_width)

    def motor_output(self):
        """
        If a button neuron's output (range [0, 1]) increases to more than or equal to 0.75,
        then its button is turned "on" and produces a "click." The button is turned "off" when
        that neuron's output falls below 0.75.
        :return: output
        """
        # Set activation threshold
        activation = [0, 0]  # Initial activation is zero
        threshold = 0.75  # Threshold for output

        # consider adding noise to output before multiplying by motor gains,
        # drawn from a Gaussian distribution with (mu=0, var=0.05)
        o7 = self.brain.Y[6] + self.brain.Theta[6]  # output of n7
        o8 = self.brain.Y[7] + self.brain.Theta[7]  # output of n8
        activation_left = expit(o7 * self.MW[0])
        activation_right = expit(o8 * self.MW[1])

        if activation_left >= threshold and not self.button_state[0]:
            activation[0] = -1   # set left activation to -1 to influence velocity to the left
            self.button_state[0] = True
        elif activation_left < threshold and self.button_state[0]:
            self.button_state[0] = False

        if activation_right >= threshold and not self.button_state[1]:
            activation[1] = 1  # set right to one to influence velocity to the right
            self.button_state[1] = True
        elif activation_right < threshold and self.button_state[1]:
            self.button_state[1] = False

        # return activation, [activation_left, activation_right]
        return activation


class DirectVelocityAgent(EmbodiedAgentV2):
    """
    This is a version of the embodied agent (v2) with direct velocity control (no button pressing).
    It needs to be paired with a DirectTracker.
    """

    def __init__(self, network, agent_parameters, screen_width):
        EmbodiedAgentV2.__init__(self, network, agent_parameters, screen_width)

    def motor_output(self):
        """
        One neuron controls leftward, the other rightward velocity.
        Each velocity is calculated by mapping the activation
        in the range of [0, 1] to the range [-1, 1] and then multiplying by output gain.
        :return: output
        """
        # consider adding noise to output before multiplying by motor gains,
        # drawn from a Gaussian distribution with (mu=0, var=0.05)
        o7 = expit(self.brain.Y[6] + self.brain.Theta[6])  # output of n7
        o8 = expit(self.brain.Y[7] + self.brain.Theta[7])  # output of n8

        # activation_left = int_linmap(o7, [0, 1], [-1, 1]) * self.MW[0]
        # activation_right = int_linmap(o8, [0, 1], [-1, 1]) * self.MW[1]

        # activation_left = o7 * self.MW[0] + o8 * self.MW[1]
        # activation_right = o7 * self.MW[2] + o8 * self.MW[3]

        # activation_left = o7 * self.MW[0]
        # activation_left = o7 * self.MW[0] * -1
        # activation_right = o8 * self.MW[1]

        activation_left = (o7 * self.MW[0] + o8 * self.MW[1]) * -1
        activation_right = o7 * self.MW[2] + o8 * self.MW[3]

        activation = [activation_left, activation_right]
        neuron_output = [o7, o8]
        return [activation, neuron_output]
