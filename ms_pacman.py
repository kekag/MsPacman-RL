import gym
import gym.wrappers
import dill
from ast import literal_eval
from concurrent import futures
from tempfile import gettempdir

import grpc
import mpm_pb2
import mpm_pb2_grpc

from model import *
from parse import *
from process import *

# gRPC service implementation
class Processor(mpm_pb2_grpc.ProcessorServicer):
    def ProcessModel(self, request, context):
        env = dill.loads(request.env)
        state = dill.loads(request.state)

        next_state, current_reward, done, info = env.step(request.action)
        next_state = np.asarray(next_state)
        next_state = next_state.reshape((1,) + next_state.shape+(1,))

        target = current_reward + gamma * np.max(model.predict(next_state))
        target_vec = model.predict(state)[0]
        target_vec[request.action] = target

        model.fit(state, target_vec.reshape(-1, action_count), epochs=1, verbose=0)

        return mpm_pb2.ModelResponse(env=dill.dumps(env),
                                     array=dill.dumps(next_state),
                                     info=str(info),
                                     reward=current_reward,
                                     done=done)

# Decide only after parsing whether to launch server, should not need to initialize training variables
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    mpm_pb2_grpc.add_ProcessorServicer_to_server(Processor(), server)
    print("Initializing server on port 2070")
    server.add_insecure_port("[::]:2070")
    server.start()
    print(f"Waiting for client to train and fit {name}.h5 model...")
    server.wait_for_termination()

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

if run_as == Run.server:
    serve()

def main():
    global env
    global model
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

            action, action_type, always_noop = get_action(always_noop, epsilon, np.argmax(model.predict(state)), env.action_space.sample())

            if run_as == Run.client:
                with grpc.insecure_channel("localhost:2070") as channel:
                    stub = mpm_pb2_grpc.ProcessorStub(channel)
                    response = stub.ProcessModel(mpm_pb2.ModelRequest(env=dill.dumps(env),
                                                                      state=dill.dumps(state),
                                                                      action=action))
                print("wow")
                env = dill.loads(response.env)
                print("wow")
                next_state = dill.loads(response.array)
                print("wow")
                info = literal_eval(response.info) # evaluates as dictionary
                print("wow")
                current_reward = response.reward
                done = response.done

            elif run_as == Run.local:
                next_state, current_reward, done, info = env.step(action)
                next_state = np.asarray(next_state)
                next_state = next_state.reshape((1,) + next_state.shape+(1,))

                # Q-value for action
                target = current_reward + gamma * np.max(model.predict(next_state))
                # Array of Q-values for all actions
                target_vec = model.predict(state)[0]
                # Change actions value to be target for fitting
                target_vec[action] = target

                model.fit(state, target_vec.reshape(-1, action_count), epochs=1, verbose=0)

            # current_reward *= 2
            total_reward += current_reward
            # total_reward += 1 # reward each survived tick

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