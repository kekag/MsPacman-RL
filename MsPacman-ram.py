import gym
import gym.wrappers
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import sys
import os

def create_model(bytes, action_count, channel_count):
    kernel_size = 3
    pool_size = 2
    input_shape = (bytes, channel_count)
    
    model = keras.models.Sequential()
    model.add(keras.layers.Conv1D(128, kernel_size=kernel_size, activation="softplus", padding="same", input_shape=input_shape))
    model.add(keras.layers.Conv1D(128, kernel_size=kernel_size, activation="softplus", padding="same", input_shape=input_shape))
    model.add(keras.layers.MaxPool1D(pool_size=pool_size))

    model.add(keras.layers.Conv1D(64, kernel_size=kernel_size, 
    activation="softplus", padding="same"))
    model.add(keras.layers.Conv1D(64, kernel_size=kernel_size, activation="softplus", padding="same", input_shape=input_shape))
    model.add(keras.layers.MaxPool1D(pool_size=pool_size))
   
    model.add(keras.layers.Conv1D(32, kernel_size=kernel_size, activation="softplus", padding="same"))
    
    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(32, activation="softplus"))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(action_count, activation="softplus"))

    model.compile(loss="mse", optimizer="adam", metrics=["mae"])

    print(model.summary())
    return model

def main():
    model_filename = "ms_pacman_ram.h5"
    n_epochs = 25
    n_bytes = 128
    env = gym.make('MsPacman-ram-v0')
    # print("Action Space:     ", env.action_space)
    print("Action Meanings:  ", env.get_action_meanings())
    # print("Action Keys:      ", env.get_keys_to_action())
    print("Action Count:     ", env.action_space.n )

    channel_count = 1
    action_count = env.action_space.n
    if os.path.exists(model_filename):
        model = keras.models.load_model(model_filename)
    else:
        print("creating model")
        model = create_model(n_bytes, action_count, channel_count)
    gamma = 0.9925
    epsilon = 0.25

    for i in range(n_epochs):
        print("epoch:", i )
        state = env.reset()
        state = np.asarray(state)
        state = state.reshape((1,)+state.shape+(1,))
        done = False
        total_reward = 0
        count = 0

        while not done:
            env.render()
           
            rand = np.random.random()
            if rand < epsilon:
                action = env.action_space.sample()
                print("random", env.get_action_meanings()[action])
            elif rand >= epsilon/3 and rand < 2*epsilon/3: 
                action = np.random.randint(2, 9)
                print("random", env.get_action_meanings()[action])
            elif rand >= 2*epsilon/3 and rand < epsilon: 
                action = np.random.randint(2, 9)
                print("random", env.get_action_meanings()[action])
            else:
                action = np.argmax(model.predict(state))
                print("predict", env.get_action_meanings()[action])

            next_state, current_reward, done, info = env.step(action)
            next_state = np.asarray(next_state)
            next_state = next_state.reshape((1,)+next_state.shape+(1,))
            current_reward *= 10
            total_reward += current_reward

            target = current_reward + gamma * np.max(model.predict(next_state))
            target_vec = model.predict(state)[ 0 ]
            target_vec[action] = target

            model.fit(state, target_vec.reshape(-1, action_count), epochs=1, verbose=0)

            state = next_state
            count += 1
            total_reward += 1
            if done:
                print(f"Reward: {total_reward}\nCount: {count}\n")

        env.render()
    env.close()
    model.save(model_filename)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard interrupt: shutting down")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)