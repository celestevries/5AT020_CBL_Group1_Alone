from gymnasium import spaces
import gymnasium as gym
import numpy as np
import scipy.signal as signal

class EnvPMSM(gym.Env):
    def __init__(self, sys_params, meas_noise = False, render_mode = None):
        # System parameters
        self.dt     = sys_params["dt"]              # Simulation step time [s]
        self.r      = sys_params["r"]               # Phase Stator Resistance [Ohm]
        self.ld     = sys_params["ld"]              # D-axis Inductance [H]
        self.lq     = sys_params["lq"]              # Q-axis Inductance [H]
        self.lambda_PM = sys_params["lambda_PM"]    # Flux-linkage due to permanent magnets [Wb]
        self.we     = sys_params["we"]              # Nominal speed [rad/s]
        self.vdc    = sys_params["vdc"]             # DC bus voltage [V]
        
        # Maximum amount of simulation steps
        self.max_steps = sys_params["sim_steps"]
        
        # Maximum voltage [V]
        self.vdq_max = self.vdc/2

        # Maximum current [A]
        self.i_max  = sys_params["i_max"]

        # Reference values
        self.id_ref = sys_params["id_ref"]
        self.iq_ref = sys_params["iq_ref"]

        # Initial values
        self.id0 = 0
        self.iq0 = 0

        # Measurement noise
        self.meas_noise = meas_noise

        # Define state-space representation
        self.__state_space()

        # Limitations for the system
        # Actions
        self.min_vd, self.max_vd = [-1.0, 1.0]
        self.min_vq, self.max_vq = [-1.0, 1.0]

        self.low_actions = np.array(
            [self.min_vd, self.min_vq], dtype=np.float32
        )
        self.high_actions = np.array(
            [self.max_vd, self.max_vq], dtype=np.float32
        )

        # Observations
        self.min_id,     self.max_id     = [-1.0, 1.0]
        self.min_iq,     self.max_iq     = [-1.0, 1.0]

        self.low_observations = np.array(
            [self.min_id, self.min_iq], dtype=np.float32
        )
        self.high_observations = np.array(
            [self.max_id, self.max_iq], dtype=np.float32
        )

        # Render mode
        self.render_mode = render_mode

        # Define action and observation space within a Box property
        self.action_space = spaces.Box(
            low=self.low_actions, high=self.high_actions, dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=self.low_observations, high=self.high_observations, dtype=np.float32
        )

    def step(self, action: np.ndarray):
        # Measurement noise
        SNR = 40 # [dB]
        noise_gain = 1/np.power(10,SNR/20)
        e = np.random.normal(0, noise_gain) if self.meas_noise else 0

        # Denormalize action
        action_vdq = self.vdq_max * action

        s_t = np.array([self.id,
                        self.iq])
        a_t = action_vdq

        # s(t+1) = ad * s(t) + bd * a(t) + w
        id_next, iq_next = self.ad @ s_t + self.bd @ a_t + self.wd
        # Rescale the current states to limit it within the boundaries if needed
        norm_idq_next = np.sqrt(np.power(id_next, 2) + np.power(iq_next, 2))
        factor_idq = self.i_max / norm_idq_next
        factor_idq = factor_idq if factor_idq < 1 else 1
        id_next, iq_next = factor_idq * np.array([id_next, iq_next])

        # Normalize observation
        id_next_norm = id_next / self.i_max
        iq_next_norm = iq_next / self.i_max
        # Observation: [id, iq]
        obs = np.array([id_next_norm + e, iq_next_norm + e], dtype=np.float32)

        terminated = True if self.steps == self.max_steps - 1 else False

        # Reward function
        id_norm = (self.id + e) / self.i_max
        iq_norm = (self.iq + e)/ self.i_max
        id_ref_norm = self.id_ref / self.i_max
        iq_ref_norm = self.iq_ref / self.i_max
        e_id = np.abs(id_norm - id_ref_norm)
        e_iq = np.abs(iq_norm - iq_ref_norm)

        reward = -(e_id + e_iq)

        # Update states
        self.id = id_next
        self.iq = iq_next
        
        # Increase simulations steps
        self.steps += 1

        return obs, reward, terminated, False, {}

    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed)
        self.steps = 0

        # Initialization of normalized dq-currents
        # [id,iq]
        id_norm = self.id0
        iq_norm = self.iq0

        # Store idq
        self.id = self.i_max * id_norm
        self.iq = self.i_max * iq_norm

        obs = np.array([id_norm, iq_norm], dtype=np.float32)

        return obs, {}
    
    def __state_space(self):
        # dq-frame continuous state-space
        # dx/dt = a*x + b*u
        # [dId/dt] = [-R/Ld      we*Lq/Ld][Id]  +  [1/Ld      0 ][Vd] + [      0      ]
        # [dIq/dt]   [-we*Ld/Lq     -R/Lq][Iq]     [ 0      1/Lq][Vq]   [-we*lambda_PM/Lq]
        a = np.array([[-self.r / self.ld,           self.we * self.lq / self.ld],
                      [-self.we * self.ld / self.lq,     -self.r / self.lq]])
        b = np.array([[1 / self.ld, 0],
                      [0, 1 / self.lq]])
        w = np.array([[0], [-self.we * self.lambda_PM/self.lq]])
        c = np.eye(2)
        d = np.zeros((2,2))

        bw = np.hstack((b, w))
        dw = np.hstack((d, np.zeros((2,1))))
        (ad, bdw, cd, _, _) = signal.cont2discrete((a, bw, c, dw), self.dt, method='zoh')

        # s_(t+1) = ad * s(t) + bd * a(t) + w
        # where ad and bd are 2x2 matrices, s(t) the state [Id, Iq], and a(t) the actions [Vd, Vq].
        # s(t) = dq currents
        # a(t) = dq voltages
        # w = disturbance due to flux-linkage from permanent magnets
        self.ad = ad
        self.bd = bdw[:,:b.shape[1]]
        self.wd = bdw[:,b.shape[1]:].squeeze()
        self.cd = cd