import gym
import gym.wrappers
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import keyboard

def plot_history(history):
    fig = plt.figure(figsize=(10, 7))
    plt.plot(history)
    plt.grid(True)
    plt.xlabel("epoch")
    plt.ylabel("reward")
    fig.savefig("plot.pdf")

def create_model(stack_count, frame_size, action_count, channel_count):
    kernel_size = (stack_count, 4, 4)
    pool_size = (2, 2, 2)
    input_shape = (stack_count, frame_size, frame_size, channel_count)

    model = keras.models.Sequential()
    model.add(keras.layers.Conv3D(256, kernel_size=kernel_size, activation="relu", padding="same", input_shape=input_shape))
    model.add(keras.layers.MaxPool3D(pool_size=pool_size))

    model.add(keras.layers.Conv3D(128, kernel_size=kernel_size, activation="relu", padding="same"))
    model.add(keras.layers.MaxPool3D(pool_size=pool_size))
    
    model.add(keras.layers.Conv3D(64, kernel_size=kernel_size, activation="relu", padding="same"))
    
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation="softplus"))

    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(action_count, activation="softplus"))

    model.compile(loss="mse", optimizer="nadam", metrics=["mae"])

    print(model.summary())
    return model

g_env = gym.make('MsPacmanNoFrameskip-v4')
model_filename = "ms_pacman.h5"

n_stack = 4
channel_count = 1
frame_size = g_env.observation_space.shape[1]
frame_size = 84
action_count = g_env.action_space.n

if os.path.exists(model_filename):
    model = keras.models.load_model(model_filename)
else:
    model = create_model(n_stack, frame_size, action_count, channel_count)

def main():
    render = True
    verbose = True
    rewards = []
    n_epochs = 1
    
    env = gym.wrappers.AtariPreprocessing(g_env, terminal_on_life_loss=False, scale_obs=False)
    env = gym.wrappers.FrameStack(env, n_stack)

    if verbose:
        # print("Action Space:     ", env.action_space)
        print("Action Meanings:  ", env.get_action_meanings())
        # print("Action Keys:      ", env.get_keys_to_action())
        print("Action Count:     ", env.action_space.n )

    gamma = 0.992
    epsilon = 0.05

    for i in range(n_epochs):
        print("epoch:", i )
        state = env.reset()
        state = np.asarray(state)
        state = state.reshape((1,)+state.shape+(1,))
        done = False
        total_reward = 0

        while not done:
            if render:
                env.render()

            rand = np.random.random()
            chosen_action = "       "
            always_noop = False
            input_action = False
            if keyboard.is_pressed("up"):
                action = 1
                input_action = True
            if keyboard.is_pressed("right"):
                action = 2
                input_action = True
            if keyboard.is_pressed("left"):
                action = 3
                input_action = True
            if keyboard.is_pressed("down"):
                action = 4
                input_action = True
            if keyboard.is_pressed("space"):
                if always_noop:
                    print("⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻")
                    print("         DISABLED ALWAYS NOOP        ")
                    print("_______________________________________")
                    always_noop = False
                else:
                    print("⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻")
                    print("         ENABLED ALWAYS NOOP        ")
                    print("_______________________________________")
                    always_noop = True
            if always_noop:
                action = 0
            elif rand < epsilon and not input_action:
                action = env.action_space.sample()
                chosen_action = " (rand)"
            elif not input_action:
                action = np.argmax(model.predict(state))
                
            if input_action:
                chosen_action = " (inpt)"
            input_action = False

            next_state, current_reward, done, info = env.step(action)
            next_state = np.asarray(next_state)
            next_state = next_state.reshape((1,) + next_state.shape+(1,))

            frame_info = "ACTION: %9s%s | REWARD: %3i | LIVES: %d" % (env.get_action_meanings()[action], chosen_action, current_reward, info.get('ale.lives'))
            if verbose:
                print(frame_info)

            current_reward *= 5
            total_reward += current_reward

            target = current_reward + gamma * np.max(model.predict(next_state))
            target_vec = model.predict(state)[0]
            target_vec[action] = target

            model.fit(state, target_vec.reshape(-1, action_count), epochs=1, verbose=0)

            state = next_state
            # total_reward += 1
            if done:
                rewards.append(total_reward)
                print("Reward:", total_reward)
                print()

        model.save(model_filename)
        if render:
            env.render()
    env.close()
    model.save(model_filename)
    plot_history(rewards)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard interrupt: shutting down")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)