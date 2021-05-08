import numpy as np
import matplotlib.pyplot as plt
import gym
import gym.wrappers
import os
import sys

# Only print tf errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow.keras as keras
import keyboard

def plot_reward(history, filename):
    fig = plt.figure(figsize=(10, 7))
    plt.plot(history)
    plt.grid(True)
    plt.xlabel("episode")
    plt.ylabel("total reward")
    # Don't overwrite any existing plots
    if os.path.exists(filename):
        unique_file = False
        i = 1
        while not unique_file:
            filename = "%s_%d" % (filename, i)
            unique_file = os.path.exists(filename)
            i += 1

    fig.savefig(filename)

def create_image_processing_model(size, n_stack, n_actions, n_channels, model_loss, model_optimizer, model_metrics):
    kernel_size = (n_stack, 4, 4)
    pool_size = (n_stack//2, 2, 2)
    input_shape = (n_stack, size, size, n_channels)

    model = keras.models.Sequential()
    model.add(keras.layers.Conv3D(128, kernel_size=kernel_size, activation="relu", padding="same", input_shape=input_shape))
    model.add(keras.layers.MaxPool3D(pool_size=pool_size))

    model.add(keras.layers.Conv3D(64, kernel_size=kernel_size, activation="relu", padding="same"))
    model.add(keras.layers.MaxPool3D(pool_size=pool_size))
    
    model.add(keras.layers.Conv3D(32, kernel_size=kernel_size, activation="relu", padding="same"))
    
    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(64, activation="softplus"))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(action_count, activation="softplus"))

    model.compile(loss=model_loss, optimizer=model_optimizer, metrics=model_metrics)

    print(model.summary())
    return model

def create_state_model(n_bytes, action_count, channel_count, model_loss, model_optimizer, model_metrics):
    kernel_size = n_bytes // 32
    pool_size = n_bytes // 64
    input_shape = (n_bytes, channel_count)
    
    model = keras.models.Sequential()
    model.add(keras.layers.Conv1D(128, kernel_size=kernel_size, activation="relu", padding="same", input_shape=input_shape))
    model.add(keras.layers.MaxPool1D(pool_size=pool_size))

    model.add(keras.layers.Conv1D(64, kernel_size=kernel_size, 
    activation="relu", padding="same"))
    model.add(keras.layers.MaxPool1D(pool_size=pool_size))
   
    model.add(keras.layers.Conv1D(32, kernel_size=kernel_size, activation="relu", padding="same"))
    
    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(64, activation="softplus", bias_initializer="he_normal"))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(action_count, activation="softplus", bias_initializer="he_normal"))

    model.compile(loss=model_loss, optimizer=model_optimizer, metrics=model_metrics)

    print(model.summary())
    return model

# Arg defaults
state = True        # Use RAM/state (True) or image stack (False) as input
gamma = 0.995       # Remember past experience 
epsilon = 0.05      # Random operation percentage
n_episodes = 5      # Number of Training iterations
one_life = False    # Terminate training after losing first life
verbose = True      # Print information each tick
render = True       # Render window

# Basic parser
if len(sys.argv) > 1:
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] in ("--image", "-image", "-i"):
            state = False
        elif args[i] in ("--gamma", "-gamma", "-g"):
            try:
                gamma = float(args[i+1])
                if gamma < 0:
                    gamma = 0
                elif gamma > 1:
                    gamma = 1
                i += 1
            except:
                print(f"Expected float after '{args[i]}' flag, skipping...")
        elif args[i] in ("--epsilon", "-epsilon", "-e"):
            try:
                epsilon = float(args[i+1])
                if epsilon < 0:
                    epsilon = 0
                elif epsilon > 1:
                    epsilon = 1
                i += 1
            except:
                print(f"Expected float after '{args[i]}' flag, skipping...")
        elif args[i] in ("--numeps", "-numeps", "-n"):
            try:
                n_episodes = int(args[i+1])
                if n_episodes < 1:
                    n_episodes = 1
                if n_episodes > 2500:
                    n_episodes = 2500
                i += 1
            except:
                print(f"Expected int after '{args[i]}' flag, skipping...")
        elif args[i] in ("--onelife", "-onelife", "-o"):
            one_life = True
        elif args[i] in ("--background", "-background", "-b"):
            verbose = False
            render = False
        else:
            print("FLAG USAGE\n")
            print("-image")
            print("\tUse image stack environment")
            print("-gamma (float)")
            print("\tValue of future reward [default 0.995]")
            print("-epsilon (float)")
            print("\tChance of random step [default 0.05]")
            print("-numeps (int)")
            print("\tNumber of training iterations [default 5]")
            print("-onelife")
            print("\tTerminate each iteration after losing first life")
            print("-background")
            print("\tHide window and display only basic information\n")
            sys.exit(0)
        i += 1

if not state:
    env = gym.make('MsPacmanNoFrameskip-v4')
    frame_stack = 4
    frame_size = 84
    env = gym.wrappers.AtariPreprocessing(env, terminal_on_life_loss=one_life, scale_obs=False)
    env = gym.wrappers.FrameStack(env, frame_stack)
    model_name = "ms_pacman.h5"
    figure_name = "ms_pacman_plot.pdf"
else:
    env = gym.make('MsPacman-ram-v0')
    # The AtariPreprocessing wrapper doesn't support RAM as obs_type, so I'll have to 
    # manually check each step to support termination on life loss for one_life
    n_bytes = 128
    model_name = "ms_pacman_ram.h5"
    figure_name = "ms_pacman_ram_plot.pdf"

model_path = "models"
model_file = os.path.join(model_path, model_name)

figure_path = "charts"
figure_file = os.path.join(figure_path, figure_name)

model_loss = "MSE"
model_optimizer = "Nadam"
model_metrics = ["MAE"]

channel_count = 1
action_count = env.action_space.n

episodes_processed = 0
rewards = []

if os.path.exists(model_file):
    model = keras.models.load_model(model_file)
else:
    if not state:
        model = create_image_processing_model(frame_size, frame_stack, action_count, channel_count, model_loss, model_optimizer, model_metrics)
    else:
        model = create_state_model(n_bytes, action_count, channel_count, model_loss, model_optimizer, model_metrics)

def main():
    global episodes_processed
    global rewards
    
    if verbose:
        print()
        print("Action Space:     ", env.action_space)
        print("Action Meanings:  ", env.get_action_meanings())
        print("Action Keys:      ", env.get_keys_to_action())
        print()

    for i in range(n_episodes):
        print("Episode:", i )
        if verbose:
            print()

        state = env.reset()
        state = np.asarray(state)
        state = state.reshape((1,)+state.shape+(1,))

        done = False
        always_noop = False
        total_reward = 0
        tick = 0

        while not done:
            if render:
                env.render()

            # Process input actions
            action = 0
            action_type = ""

            if keyboard.is_pressed("space"):
                if always_noop:
                    if verbose:
                        print("⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻")
                        print("               DISABLED ALWAYS NOOP              ")
                        print("_________________________________________________")
                    always_noop = False
                else:
                    if verbose:
                        print("⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻")
                        print("               ENABLED ALWAYS NOOP               ")
                        print("_________________________________________________")
                    always_noop = True
            elif keyboard.is_pressed("up") and keyboard.is_pressed("right"):
                action = 5
            elif keyboard.is_pressed("up") and keyboard.is_pressed("left"):
                action = 6
            elif keyboard.is_pressed("down") and keyboard.is_pressed("right"):
                action = 7
            elif keyboard.is_pressed("down") and keyboard.is_pressed("left"):
                action = 8
            elif keyboard.is_pressed("up"):
                action = 1
            elif keyboard.is_pressed("right"):
                action = 2
            elif keyboard.is_pressed("left"):
                action = 3
            elif keyboard.is_pressed("down"):
                action = 4

            if action != 0:
                action_type = "(inpt)"
            else:
                # Process predicted or random environment actions
                rand = np.random.random()
                if always_noop:
                    action = 0
                elif rand < epsilon:
                    action = env.action_space.sample()
                    action_type = "(rand)"
                else:
                    action = np.argmax(model.predict(state))

            next_state, current_reward, done, info = env.step(action)
            next_state = np.asarray(next_state)
            next_state = next_state.reshape((1,) + next_state.shape+(1,))

            frame_info = "EP %i. ACTION: %9s%7s | REWARD: %4i | LIVES: %d" % (episodes_processed, env.get_action_meanings()[action], action_type, current_reward, info.get('ale.lives'))
            if verbose:
                print(frame_info)

            # current_reward *= 2
            total_reward += current_reward
            # total_reward += 1 # reward each survived tick

            target = current_reward + gamma * np.max(model.predict(next_state))
            target_vec = model.predict(state)[0]
            target_vec[action] = target

            model.fit(state, target_vec.reshape(-1, action_count), epochs=1, verbose=0)

            state = next_state
           
            if done:
                rewards.append(total_reward)
                if verbose:
                    print()
                print("Reward:", total_reward)
                print()

            tick += 1

        model.save(model_file)
        if render:
            env.render()

        episodes_processed += 1

    env.close()
    model.save(model_file)
    if n_episodes >= 5:
        plot_reward(rewards, figure_file)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nKEYBOARD INTERRUPT")
        try:
            save = input("Save model data? [y/n] ")
            if save == 'y':
                env.close()
                model.save(model_file)
                if episodes_processed >= 5:
                    plot_reward(rewards, figure_file)
            sys.exit(0)
        except SystemExit:
            os._exit(0)