import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.core.magic import register_cell_magic
from IPython.display import clear_output

# Column names
columns = pd.MultiIndex.from_tuples([
            ("Error", "Absolute [A]"),
            ("Error", "Relative [%]"),
            ("Error", "SSE"),
            ("", "Control energy"),
            ("Settling time [ms]", "Id"),
            ("Settling time [ms]", "Iq"),
            ("Overshoot [%]", "Id"),
            ("Overshoot [%]", "Iq "),
            ("Undershoot [%]", "Id"),
            ("Undershoot [%]", "Iq"),
            ("Computation time [ms]", "Min"),
            ("Computation time [ms]", "Max"),
            ("Computation time [ms]", "Avg")])

class Metrics:
    def __init__(self, idq_ref, dt):
        self.dt = dt                        # Simulation step time [s]
        self.idq_ref = np.array(idq_ref)    # Reference currents in dq-axis [A]

        # Function for matrix-vector substraction and division
        self.mv_sub = lambda matrix, vector: (matrix.transpose() - vector).transpose()
        self.mv_div = lambda matrix, vector: (matrix.transpose() / vector).transpose()


    def error_r(self, idq):
        return 1/5*np.sum(self.mv_div(np.abs(self.mv_sub(idq[:,-5:], self.idq_ref)), self.idq_ref))
    
    def error(self, idq):
        return 1/5*np.sum(np.abs(self.mv_sub(idq[:,-5:], self.idq_ref)))
    
    def sse(self, idq):
        return np.sum(self.mv_sub(idq, self.idq_ref)**2)
    
    def control_energy(self, vdq):
        return self.dt * np.sum(vdq**2)
    
    # def settling_time(self, idq, sim_plot_steps):
    #     # Define tolerance band
    #     tolerance = 0.02
    #     tolerance = 0.1
    #     tolerance_band = np.abs(tolerance * self.idq_ref)

    #     # Utilize the flipped version if idq and calculate its error
    #     idq_flip = np.flip(idq, 1)
    #     error = np.abs(self.mv_sub(idq_flip, self.idq_ref))

    #     # Search for the index where the error gets out of bounds
    #     index = sim_plot_steps - np.argwhere(error.transpose() >= tolerance_band)[0:2][:,0]
        
    #     # Utilize the index to return the settling time
    #     time = self.dt * index
        
    #     return time
    
    def settling_time(self, time_vec, signal, stability_threshold=0.05, window_percent=10):
        """
        Estimates the settling time for one or more signals by analyzing the
        stability of their moving average and moving standard deviation.

        Args:
            time_vec (np.ndarray): The 1D time vector corresponding to the signals.
            signal (np.ndarray): The signal data. Can be a 1D array for a single
                                signal, or a 2D array where each column is a
                                separate signal.
            stability_threshold (float): The threshold for the rate of change of
                                        the statistics. A smaller value means a
                                        stricter definition of "stable".
            window_percent (int): The size of the moving window as a percentage
                                of the total signal length.

        Returns:
            tuple: A tuple containing:
                - np.ndarray: An array of estimated settling times (one per channel).
                            Contains None for channels that never settle.
                - list: A list of dictionaries, each containing the visualization
                        data ('moving_avg', 'moving_std', 'settling_index') for
                        the corresponding channel.
        """
        if len(signal.shape) == 1:
            # If a 1D array is passed, reshape it to a 2D array with one column
            signal = signal.reshape(-1, 1)

        if signal.shape[1] != len(time_vec):
            raise ValueError("Signal and time vector must have the same number of samples (rows).")

        n_channels = signal.shape[0]
        settling_times = []
        viz_data_list = []

        # --- Iterate over each channel (column) in the signal array ---
        for i in range(n_channels):
            current_signal = signal[i, :]
            s = pd.Series(current_signal)

            # --- 1. Calculate window size ---
            window_size = int((window_percent / 100) * len(s))
            if window_size < 2:
                window_size = 2

            # --- 2. Calculate moving statistics ---
            moving_avg = s.rolling(window=window_size, center=True).mean()
            moving_std = s.rolling(window=window_size, center=True).std()
            
            moving_avg.fillna(method='bfill', inplace=True)
            moving_avg.fillna(method='ffill', inplace=True)
            moving_std.fillna(method='bfill', inplace=True)
            moving_std.fillna(method='ffill', inplace=True)

            # --- 3. Calculate the rate of change ---
            signal_abs_mean = np.mean(np.abs(current_signal))
            if signal_abs_mean < 1e-9: signal_abs_mean = 1.0

            avg_rate_of_change = np.abs(np.diff(moving_avg) / signal_abs_mean)
            std_rate_of_change = np.abs(np.diff(moving_std) / signal_abs_mean)

            # --- 4. Find where the signal becomes stable ---
            is_unstable = (avg_rate_of_change > stability_threshold) | \
                        (std_rate_of_change > stability_threshold)
            
            unstable_indices = np.where(is_unstable)[0]

            if len(unstable_indices) == 0:
                settling_index = window_size
            else:
                settling_index = unstable_indices[-1] + 1
            
            if settling_index >= len(time_vec):
                settling_time = None
            else:
                settling_time = time_vec[settling_index]
            
            settling_times.append(settling_time)
            
            viz_data = {
                'moving_avg': moving_avg.to_numpy(),
                'moving_std': moving_std.to_numpy(),
                'settling_index': settling_index
            }
            viz_data_list.append(viz_data)

        return np.array(settling_times), viz_data_list

    def overshoot(self, idq):
        return np.array([100*np.abs((np.max(i)-iref)/iref) if np.abs(np.max(i)) > np.abs(iref) else np.nan for i, iref in zip(idq, self.idq_ref)])
    
    def undershoot(self, idq):
        return np.array([100*np.abs((np.min(i)-iref)/iref) if np.abs(np.min(i)) < np.abs(i0) else np.nan for i, i0, iref in zip(idq, idq[:,0], self.idq_ref)])
    
    def computation_time(self, controller, controller_inputs):
        start_time = time.perf_counter()
        action = controller(**controller_inputs)
        elapsed_time_s = time.perf_counter() - start_time
        elapsed_time_ms = 1e3*elapsed_time_s

        return action, elapsed_time_ms

@register_cell_magic
def skip(line, cell):
    return

def plot_results(subplot, env, time, computation_time_ms, id_history, iq_history, vd_history, vq_history):
    rows, cols, col_index = subplot

    fignum = [1 + col_index + r * cols for r in range(rows)]

    plt.subplot(rows, cols, fignum[0])
    plt.plot(time*1000, id_history, label='id')
    plt.plot(time*1000, np.ones_like(time)*env.id_ref, '--', label='id_ref')
    plt.plot(time*1000, iq_history, label='iq')
    plt.plot(time*1000, np.ones_like(time)*env.iq_ref, '--', label='iq_ref')
    plt.grid(True)
    plt.legend()
    plt.xlabel('Time [ms]')
    plt.ylabel('Current [A]')
    plt.title('Current dynamic behaviour')
    
    plt.subplot(rows, cols, fignum[1])
    plt.plot(time*1000, vd_history, label='vd')
    plt.plot(time*1000, vq_history, label='vq')
    plt.grid(True)
    plt.legend()
    plt.xlabel('Time [ms]')
    plt.ylabel('Voltage [V]')
    plt.title('Control Voltages')
    
    plt.subplot(rows, cols, fignum[2])
    plt.plot(time*1000, computation_time_ms, "-o")
    plt.grid(True)
    plt.xlabel('Time [ms]')
    plt.ylabel('Computation time [ms]')
    plt.title('Computation time in each simulation step')

    plt.tight_layout()

def decouple(env):
    Vd_ff = -env.we * env.lq * env.iq
    Vq_ff = env.we * (env.ld * env.id + env.lambda_PM)
    return np.array([Vd_ff, Vq_ff])

def run_simulation(envs, controller, controller_type, metrics, figsize, sim_steps=100, plot=True):
    if controller_type not in ["PI", "MPC", "QL", "SB3", "RL"]:
        print("Unavailable controller type")
        return -1

    envs = envs if type(envs) is list else [envs]
    table_data = []

    if plot:
        plt.figure(figsize=figsize)

    for idx, env in enumerate(envs):
        # Create list to store multiple table data

        if controller_type == "SB3":
            if hasattr(env, 'unwrapped'):
                base_env = env.unwrapped.envs[0]
                while hasattr(base_env, 'env'):
                    base_env = base_env.env
            else:
                base_env = env
        else:
            base_env = env.unwrapped

        # Reference currents
        id_ref = base_env.id_ref
        iq_ref = base_env.iq_ref
        
        # Check if sim_steps is greater than the maximum number of steps
        if sim_steps > base_env.max_steps:
            sim_steps = base_env.max_steps

        # Storage for plotting
        time = np.arange(sim_steps) * base_env.dt
        computation_time_ms = np.zeros(sim_steps)
        id_history = np.zeros(sim_steps)
        iq_history = np.zeros(sim_steps)
        vd_history = np.zeros(sim_steps)
        vq_history = np.zeros(sim_steps)
        
        # Run simulation
        if controller_type == "SB3":
            state_norm = env.reset()
        else:
            state_norm, _ = env.reset()
        
        step = 0
        done = False
        while not done and step < sim_steps:

            reference = np.array([id_ref, iq_ref])
            state = state_norm * base_env.i_max

            if controller_type == "QL":
                discrete_action = controller.greedy_policy(discrete_state)
                # Apply action and get new state
                discrete_state, reward, terminated, truncated, _ = env.step(discrete_action)
                action_norm = env.get_continuous_action()
                state_norm = env.get_continuous_state()
                state = state_norm * env.i_max
                action = action_norm * env.vdq_max

            if controller_type == "SB3":
                # Apply action and get new state
                controller_inputs = {"observation": state_norm}
                (action_norm, _states), comp_time_ms = metrics.computation_time(controller.predict, controller_inputs)
                # action_norm, _states = controller.predict(state_norm)
                state_norm, _, done , _ = env.step(action_norm)
                state = state_norm.flatten() * base_env.i_max
                action = action_norm.flatten() * base_env.vdq_max

            if controller_type == "RL":
                # Apply action and get new state
                controller_inputs = {"observation": state_norm}
                action_norm, comp_time_ms = metrics.computation_time(controller.select_action, controller_inputs)
                # action_norm, _states = controller.predict(state_norm)
                state_norm, _, terminated, truncated , _= env.step(action_norm)
                state = state_norm.flatten() * base_env.i_max
                action = action_norm.flatten() * base_env.vdq_max

            if controller_type == "PI":
                controller_inputs = {"reference": reference, "measured": state}
                action, comp_time_ms = metrics.computation_time(controller.control, controller_inputs)
                # action = controller.control(reference, state)
                action += decouple(env=base_env)
                action_norm = action / base_env.vdq_max
                # Apply action and get new state
                state_norm, _, terminated, truncated , _ = env.step(action_norm)
                state = state_norm * base_env.i_max

            elif controller_type == "MPC":
                controller_inputs = {"x": state, "y": state, "yref": reference}
                action, comp_time_ms = metrics.computation_time(controller.compute_input, controller_inputs)
                # action = controller.compute_input(x=state, y=state, yref=reference)
                action_norm = action / base_env.vdq_max
                # Apply action and get new state
                state_norm, _, terminated, truncated , _ = env.step(action_norm)
                state = state_norm * base_env.i_max

            # Store data
            computation_time_ms[step] = comp_time_ms
            id_history[step] = state[0]
            iq_history[step] = state[1]
            vd_history[step] = action[0]
            vq_history[step] = action[1]

            if controller_type != "SB3":
                done = terminated or truncated

            if done or step == sim_steps:
                break

            step += 1
        
        if plot:
            plot_results((3,len(envs),idx), base_env, time, computation_time_ms, id_history, iq_history, vd_history, vq_history)

        idq = np.array([id_history, iq_history])
        vdq = np.array([vd_history, vq_history])

        # Metrics
        error_r         = metrics.error_r(idq)
        error           = metrics.error(idq)
        sse             = metrics.sse(idq)
        control_energy  = metrics.control_energy(vdq)
        settling_time, _ = metrics.settling_time(time, idq)
        settling_time_ms= 1e3*settling_time    # [ms]
        overshoot       = metrics.overshoot(idq)            # dq
        undershoot      = metrics.undershoot(idq)           # dq
        min_computation_time = np.min(computation_time_ms)
        max_computation_time = np.max(computation_time_ms)
        avg_computation_time = np.mean(computation_time_ms)

        table_data.append([
            error_r, 
            error, 
            sse, 
            control_energy, 
            *settling_time_ms, 
            *overshoot, 
            *undershoot, 
            min_computation_time, 
            max_computation_time, 
            avg_computation_time
        ])

    if plot:
        plt.show()

    return table_data

class Logger:
    "Used to track episode lengths, returns, and total steps."
    def __init__(self, total_steps: int):
        self.current_step = 0
        self.current_episode = 1
        self.current_return = 0.0
        self.current_length = 0
        self.episode_returns = []
        self.episode_lengths = []
        self.episodes_history = []
        self.rewards_history = []

        self.custom_logs = {}
        self.custom_log_keys = []
        self.start_time = time.time()
        self.total_steps = total_steps
        self.header_printed = False
        
        # Logger settings
        self.log_interval = 100  # Print logs every log_interval timesteps
        self.window = 20         # Use this many items from recent logs
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], "o-")   # persistent line

    
    def log(self, reward: float, termination: bool, truncation: bool, **kwargs):
        "Updates logger with latest rewards, done flags and any custom logs."
        self.current_step += 1
        self.current_return += reward
        self.current_length += 1

        # Update tracked statistics
        if termination or truncation:
            self.episode_returns.append(self.current_return)
            self.episode_lengths.append(self.current_length)
            self.current_episode += 1
            self.current_return = 0.0
            self.current_length = 0

        # Update custom_logs with any additional keyword arguments
        for key, value in kwargs.items():
            if key not in self.custom_log_keys:
                self.custom_log_keys.append(key)
            self.custom_logs[key] = value

    def print_logs(self):
        "Prints training progress with headers and updates."
        if self.current_step % self.log_interval == 0 and len(self.episode_returns) > 0:
            elapsed_time = time.time() - self.start_time

            # Calculate other metrics
            progress = 100 * self.current_step / self.total_steps
            mean_reward = np.mean(
                self.episode_returns[-self.window:]
            ) if len(self.episode_returns) >= self.window else np.mean(self.episode_returns)
            mean_ep_length = np.mean(
                self.episode_lengths[-self.window:]
            ) if len(self.episode_lengths) >= self.window else np.mean(self.episode_lengths)
            
            # Store relevant data for plot
            self.episodes_history.append(self.current_episode)
            self.rewards_history.append(mean_reward)

            # Format elapsed time into hh:mm:ss
            hours, remainder = divmod(int(elapsed_time), 3600)
            minutes, seconds = divmod(remainder, 60)
            formatted_time = f"{hours:02}:{minutes:02}:{seconds:02}"

            # if not self.header_printed:
            log_header = (
                f"{'Progress':>8}  |  "
                f"{'Step':>8}  |  "
                f"{'Episode':>8}  |  "
                f"{'Mean Rew':>8}  |  "
                f"{'Mean Len':<7}  |  "
                f"{'Time':>8}"
            )
            # Append custom log headers
            for key in self.custom_log_keys:
                log_header += f"  |  {key:>{len(key)}}"
            print(log_header)
            self.header_printed = True

            log_string = (
                f"{progress:>7.1f}%  |  "
                f"{self.current_step:>8,}  |  "
                f"{self.current_episode:>8,}  |  "
                f"{mean_reward:>8.2f}  |  "
                f"{mean_ep_length:>8.1f}  |  "
                f"{formatted_time:>8}"
            )
            # Append custom log values
            for key in self.custom_log_keys:
                value = self.custom_logs.get(key, 0)
                # Format based on the type of value
                if isinstance(value, float):
                    log_string += f"  |  {value:>{len(key)}.2f}"
                elif isinstance(value, int):
                    log_string += f"  |  {value:>{len(key)}d}"
                else:
                    log_string += f"  |  {str(value):>{len(key)}}"
            print(f"\r{log_string}", end='')

    def plot_logs(self):
        if self.current_step % self.log_interval == 0 and len(self.episode_returns) > 0:
            mean_reward = np.mean(
                self.episode_returns[-self.window:]
            ) if len(self.episode_returns) >= self.window else np.mean(self.episode_returns)
            
            # Store relevant data for plot
            self.episodes_history.append(self.current_episode)
            self.rewards_history.append(mean_reward)
        
            # clear_output(wait=True)
            plt.figure(figsize=(10, 6))
            plt.plot(self.episodes_history, self.rewards_history)
            plt.title("Mean reward while Training plot")
            plt.grid(True)
            plt.xlabel('Episode')
            plt.legend(loc='center left') # the plot evolves to the right
            plt.show()

    @property
    def logs(self):
        return  {
            'total_steps': self.current_step,
            'total_episodes': self.current_episode - 1,
            'episode_returns': self.episode_returns,
            'episode_lengths': self.episode_lengths,
            'best_reward': np.max(self.episode_returns) if len(self.episode_returns) > 0 else None,
            'total_duration': time.time() - self.start_time,
            'mean_fps': self.current_step / (time.time() - self.start_time + 1e-6),
            'custom_logs': self.custom_logs
        }
    
class Logger1:
    def __init__(self, total_steps: int):
        self.window = 20
        self.log_interval = 1
        self.current_step = 0
        self.current_episode = 0
        self.episode_returns = []

        # --- create figure ONCE here ---
        if not hasattr(Logger, "fig"):   # static class var to prevent new figs
            Logger.fig, Logger.ax = plt.subplots()
            Logger.line, = Logger.ax.plot([], [], "o-")
            Logger.episodes_history = []
            Logger.rewards_history = []

        # give instance refs
        self.fig = Logger.fig
        self.ax = Logger.ax
        self.line = Logger.line
        self.episodes_history = Logger.episodes_history
        self.rewards_history = Logger.rewards_history
        plt.show()

    def log(self, reward: float, termination: bool, truncation: bool, **kwargs):
        self.episode_returns.append(reward)
        self.current_episode += 1
        self.current_step += 1
        # self.plot_logs()

    def plot_logs(self):
        if self.current_step % self.log_interval == 0:
            clear_output(wait=True)
            mean_reward = np.mean(
                self.episode_returns[-self.window:]
            ) if len(self.episode_returns) >= self.window else np.mean(self.episode_returns)

            # update histories
            self.episodes_history.append(self.current_episode)
            self.rewards_history.append(mean_reward)

            # update line
            self.live_plot(self.episodes_history, self.rewards_history)
    
    def live_plot(self, x, y, figsize=(7,5), title=''):
        clear_output(wait=True)
        plt.figure(figsize=figsize)
        plt.plot(x, y)
        plt.title(title)
        plt.grid(True)
        plt.xlabel('Episode')
        plt.legend(loc='center left') # the plot evolves to the right
        plt.show()