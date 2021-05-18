import numpy as np
import matplotlib.pyplot as plt
import gym
import gym.wrappers
import os
import sys
import keyboard
import enum
from concurrent import futures

# Only print tensorflow errors
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

import tensorflow as tf
import tensorflow.keras as keras

import grpc
import mpm_pb2
import mpm_pb2_grpc

Run = enum.Enum("Run", "local server client")

# Argument defaults
run_as = Run.local # Local, server, or client (could use enum, not necessary)
state = True       # Use RAM/gamestate (True) or image stack (False) as input
gamma = 0.995      # Remember past experience 
epsilon = 0.15     # Random operation percentage
n_episodes = 5     # Number of Training iterations
one_life = False   # Terminate training after losing first life
verbose = True     # Print information each tick
render = True      # Render window
decay = 0.99       # Epsilon decay rate
min_epsilon = 0.01 # Floor of decay
name = ''          # Model and figure filenames

# Basic commmand-line arg parser
if len(sys.argv) > 1:
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] in ("--local", "-local", "-l"):
            run_as = Run.local
        elif args[i] in ("--server", "-server", "-s"):
            run_as = Run.server
        elif args[i] in ("--client", "-client", "-c"):
            run_as = Run.client
        elif args[i] in ("--image", "-image", "-i"):
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
                elif n_episodes > 2500:
                    n_episodes = 2500
                i += 1
            except:
                print(f"Expected int after '{args[i]}' flag, skipping...")
        elif args[i] in ("--onelife", "-onelife", "-o"):
            one_life = True
        elif args[i] in ("--background", "-background", "-b"):
            verbose = False
            render = False
        elif args[i] in ("--decay", "-decay", "-d"):
            try:
                decay = float(args[i+1])
                if decay > 1:
                    decay = 1
                elif decay < 0.01:
                    decay = 0.01
                i += 1
            except:
                print(f"Expected float after '{args[i]}' flag, skipping...")
        elif args[i] in ("--floor", "-floor", "-f"):
            try:
                min_epsilon = float(args[i+1])
                if min_epsilon > epsilon:
                    min_epsilon = epsilon
                elif min_epsilon < 0:
                    min_epsilon = 0
                i += 1
            except:
                print(f"Expected float after '{args[i]}' flag, skipping...")
        elif args[i] in ("--name", "-name"):
            try:
                min_epsilon = args[i+1]
                i += 1
            except:
                print(f"Expected string after '{args[i]}' flag, skipping...")
        else:
            print("FLAG USAGE")
            print("-local, -server, or -client\n\tRun type enumeration [default local]")
            print("-image\n\tUse image stack environment")
            print("-gamma (float)\n\tValue of future reward [default 0.995]")
            print("-epsilon (float)\n\tValue of experimentation [default 0.15]")
            print("-numeps (int)\n\tNumber of training episodes [default 5]")
            print("-onelife\n\tTerminate each iteration after losing first life")
            print("-background\n\tHide window and display only basic information")
            print("-decay (float)\n\tEpsilon decay, per episode [default 0.98]")
            print("-floor (float)\n\tFloor of decay [default 0.01]")
            print("-name (string)\n\tModel and figure filenames, single string\n")
            sys.exit(0)
        i += 1

#/ MODELS /#
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

# gRPC service implementation
class Processor(mpm_pb2_grpc.ProcessorServicer):
    def ProcessModel(self, request, context):
        return mpm_pb2.ModelResponse()

# Decide only after parsing whether to launch server, should not need to initialize training variables
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    mpm_pb2_grpc.add_ProcessorServicer_to_server(Processor, server)
    server.add_insecure_port("[::]:2070")
    server.start()
    server.wait_for_termination()

if run_as == Run.server:
    serve()

# If running as local or client, set global model variables for RAM or image
if not state:
    env = gym.make("MsPacmanNoFrameskip-v4")
    frame_stack = 4
    frame_size = 84
    env = gym.wrappers.AtariPreprocessing(env, terminal_on_life_loss=one_life, scale_obs=False)
    env = gym.wrappers.FrameStack(env, frame_stack)
    if name == '': # Set default name
        name = "ms_pacman"
else:
    env = gym.make("MsPacman-ram-v0")
    # The AtariPreprocessing wrapper doesn't support RAM as obs_type, so I'll have to 
    # manually check each step to support termination on life loss for one_life
    n_bytes = 128
    if name == '':
        name = "ms_pacman_ram"

model_path = "models"
model_file = os.path.join(model_path, f"{name}.h5")
figure_path = "charts"
figure_file = os.path.join(figure_path, f"{name}_plot")

model_loss = "MSE"
model_optimizer = "Nadam"
model_metrics = ["MAE"]

channel_count = 1
action_count = env.action_space.n

# Use existing model, otherwise create a fresh one for either RAM or image
if os.path.exists(model_file):
    model = keras.models.load_model(model_file)
else:
    if not state:
        model = create_image_processing_model(frame_size, frame_stack, action_count, channel_count, model_loss, model_optimizer, model_metrics)
    else:
        model = create_state_model(n_bytes, action_count, channel_count, model_loss, model_optimizer, model_metrics)

#/ MATPLOTLIB PLOTTER /#
def plot_reward(history, filename):
    fig = plt.figure(figsize=(10, 7))
    plt.plot(history)
    plt.grid(True)
    plt.xlabel("episode")
    plt.ylabel("total reward")
    # Don't overwrite any existing plots
    fn = ''
    if os.path.exists(f"{filename}.pdf"):
        override = input(f"'{filename}.pdf' already exists, would you like to override it? [y/n] ")
        if override == 'y' or override == 'Y':
            fig.savefig(f"{filename}.pdf")
            return
        file_exists = True
        i = 1
        # Create new copy with name_1.pdf up to _20.pdf before overriding anyway
        while file_exists:
            if i > 20:
                break
            fn = "%s_%d.pdf" % (filename, i) 
            file_exists = os.path.exists(fn)
            i += 1
    else:
        fig.savefig(f"{filename}.pdf")
        return

    fig.savefig(fn)

episodes_processed = 0
rewards = []

#/ PROCESS ACTIONS /#
def get_action(always_noop, this_state):
    action = 0
    action_type = ''

    if keyboard.is_pressed("space"):
        if always_noop:
            print("—————————————————————————————————————————————————")
            print("               DISABLED ALWAYS NOOP              ")
            print("—————————————————————————————————————————————————")
            always_noop = False
        else:
            print("—————————————————————————————————————————————————")
            print("               ENABLED ALWAYS NOOP               ")
            print("—————————————————————————————————————————————————")
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
            action = np.argmax(model.predict(this_state))

    return action, action_type, always_noop

def main():
    global epsilon
    global episodes_processed
    global rewards
    
    if verbose:
        print("\nAction Space:     ", env.action_space)
        print("Action Meanings:  ", env.get_action_meanings())
        print("Action Keys:      \n", env.get_keys_to_action())

    for i in range(n_episodes):
        print("Episode:", i )
        if verbose:
            print()

        state = env.reset()
        state = np.asarray(state)
        state = state.reshape((1,) + state.shape+(1,))

        done = False
        total_reward = 0
        tick = 0
        always_noop = False

        while not done:
            if render:
                env.render()

            action, action_type, always_noop = get_action(always_noop, state)

            # ↑ ↑ CLIENT INPUT AND RENDERING ↑ ↑
            #     request: model, env, state, action, total_reward, gamma
            # ↓ ↓ SERVER MODEL PROCESSING  ↓ ↓ ↓
            if run_as == Run.client:
                with grpc.insecure_channel("localhost:2070") as channel:
                    stub = mpm_pb2_grpc.ProcessorStub(channel)
                    response = stub.ProcessModel(mpm_pb2.ModelRequest(seed=env.seed(),
                                                                      model='''pickled model''',   
                                                                      b64array=state,
                                                                      size=len(np.asarray(state)),
                                                                      action=action,
                                                                      reward=total_reward,
                                                                      gamma=gamma,))
            elif run_as == Run.local:
                next_state, current_reward, done, info = env.step(action)
                next_state = np.asarray(next_state)
                next_state = next_state.reshape((1,) + next_state.shape+(1,))

                # current_reward *= 2
                total_reward += current_reward
                # total_reward += 1 # reward each survived tick

                # Q-value for action
                target = current_reward + gamma * np.max(model.predict(next_state))
                # Array of Q-values for all actions
                target_vec = model.predict(state)[0]
                # Change actions value to be target for fitting
                target_vec[action] = target

                model.fit(state, target_vec.reshape(-1, action_count), epochs=1, verbose=0)

            # ↑ ↑ END SERVER STEP AND FIT ↑ ↑ ↑
            #     response: model, env, next_state, current reward, done, info
            # ↓ ↓ CLIENT OUTPUT ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓

            if verbose:
                print("EP %i. ACTION: %9s%7s | REWARD: %4i | LIVES: %d" % (episodes_processed, env.get_action_meanings()[action], action_type, current_reward, info.get('ale.lives')))

            state = next_state
            if done:
                rewards.append(total_reward)
                if verbose:
                    print()
                print(f"Reward: {total_reward}\n")

            tick += 1

        model.save(model_file)
        if render:
            env.render()

        episodes_processed += 1
        if epsilon > min_epsilon:
            epsilon *= decay

    env.close()
    model.save(model_file)
    if n_episodes >= 3:
        plot_reward(rewards, figure_file)

try:
    main()
except KeyboardInterrupt:
    print("\nKEYBOARD INTERRUPT")
    try:
        if episodes_processed > 0:
            save = input("Save model data? [y/n] ")
            if save == 'y' or save == 'Y':
                env.close()
                model.save(model_file)
                if episodes_processed >= 3:
                    plot_reward(rewards, figure_file)
        sys.exit(0)
    except SystemExit:
        os._exit(0)