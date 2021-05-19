import pickle
import gym
import gym.wrappers
from concurrent import futures

import grpc
import mpm_pb2
import mpm_pb2_grpc

from model import *
from parse import *
from process import *

server_fits = 0

# gRPC service implementation
class Processor(mpm_pb2_grpc.ProcessorServicer):
    def PredictAction(self, request, context):            
        state = pickle.loads(request.state)
        action = np.argmax(model.predict(state))
        if verbose:
            print(f"Predicting {action_meanings[action]}")
        return mpm_pb2.ActionResponse(action=action)

    def ProcessModel(self, request, context):
        global server_fits
        if server_fits == 0:
            print("Client found, fitting model")
        elif server_fits % 100 == 0 and not verbose: # sanity check every 100 fits
            print(f"{server_fits}th fit")

        state = pickle.loads(request.state)
        next_state = pickle.loads(request.next_state)

        target = request.reward + gamma * np.max(model.predict(next_state))
        target_vec = model.predict(state)[0]
        target_vec[request.action] = target

        model.fit(state, target_vec.reshape(-1, action_count), epochs=1, verbose=0)
        server_fits += 1

        if request.done:
            print("Done fitting model for current episode")
        return mpm_pb2.Empty()

    def SaveModel(self, request, context):
        print(f"Saving model to {model_file}")
        model.save(model_file)
        if not request.model_only:
            if episodes_processed >= 3:
                print(f"Saving reward plot to {figure_file}")
                plot_reward(rewards, figure_file)
        return mpm_pb2.Empty()

    def DropClient(self, request, context):
        print("Client terminated training, waiting for new client")
        return mpm_pb2.Empty()

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
if not RAMstate:
    if run_as != Run.server: # Server should not access ALE
        env = gym.make("MsPacmanNoFrameskip-v4")
        env = gym.wrappers.AtariPreprocessing(env, terminal_on_life_loss=one_life, scale_obs=False)
        env = gym.wrappers.FrameStack(env, 4)
    if name == '': # Set default name
        name = "ms_pacman"
else:
    if run_as != Run.server: 
        env = gym.make("MsPacman-ram-v0")
        # The AtariPreprocessing wrapper doesn't support RAM as obs_type, so I'll have to manually check each step to support termination on life loss for one_life
    if name == '':
        name = "ms_pacman_ram"

if run_as != Run.client: # Client should not access model
    model_path = "models"
    model_file = os.path.join(model_path, f"{name}.h5")
    figure_path = "charts"
    figure_file = os.path.join(figure_path, f"{name}_plot")

    model_loss = "MSE"
    model_optimizer = "Nadam"
    model_metrics = ["MAE"]

    channel_count = 1
    action_meanings = ['NOOP', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT', 'DOWNRIGHT', 'DOWNLEFT'] # env.get_action_meanings()
    action_count = 9 # env.action_space.n

    # Use existing model, otherwise create a fresh one for either RAM or image
    if os.path.exists(model_file):
        model = keras.models.load_model(model_file)
    else:
        print(model_file)
        if not RAMstate: # 84 => frame size, 4 => frame stack (depth of 3rd dimension)
            model = create_image_processing_model(84, 4, action_count, channel_count, model_loss, model_optimizer, model_metrics)
        else: # 128 => bytes of RAM 
            model = create_state_model(128, action_count, channel_count, model_loss, model_optimizer, model_metrics)

if run_as == Run.server:
    serve()
elif run_as == Run.client:
    stub = mpm_pb2_grpc.ProcessorStub(grpc.insecure_channel("localhost:2070"))

def main():
    global env
    global model
    global epsilon
    global episodes_processed
    global rewards
    
    if verbose:
        print("\nAction Space:     ", env.action_space)
        print("Action Meanings:  \n", env.get_action_meanings())
        # print("Action Keys:      \n", env.get_keys_to_action())

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

            action, action_type, always_noop = get_action(always_noop, epsilon, env.action_space.sample())

            if run_as == Run.client:
                if action == -1:
                    action_response = stub.PredictAction(mpm_pb2.StateRequest(state=pickle.dumps(state)))
                    action = action_response.action
                    
                next_state, current_reward, done, info = env.step(action)
                next_state = np.asarray(next_state)
                next_state = next_state.reshape((1,) + next_state.shape+(1,))

                stub.ProcessModel(mpm_pb2.ModelRequest(state=pickle.dumps(state),
                                                       next_state=pickle.dumps(next_state),
                                                       reward=current_reward,
                                                       done=done))

            elif run_as == Run.local:
                if action == -1:
                    action = np.argmax(model.predict(state))

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

            total_reward += current_reward
            # total_reward += 1 # Reward each survived tick

            if verbose:
                print("EP %i. ACTION: %9s%7s | REWARD: %4i | LIVES: %d" % (episodes_processed, env.get_action_meanings()[action], action_type, current_reward, info.get('ale.lives')))

            state = next_state
            if done:
                rewards.append(total_reward)
                if verbose:
                    print()
                print(f"Reward: {total_reward}\n")

            tick += 1
        if run_as == Run.local:
            model.save(model_file)
        else:
            stub.SaveModel(mpm_pb2.SaveRequest(model_only=True))

        if render:
            env.render()

        episodes_processed += 1
        if epsilon > min_epsilon:
            epsilon *= decay

    env.close()
    if run_as == Run.local:
        model.save(model_file)
        if n_episodes >= 3:
            plot_reward(rewards, figure_file)
    elif run_as == Run.client:
        stub.SaveModel(mpm_pb2.SaveRequest(model_only=False))

try:
    main()
except KeyboardInterrupt:
    print("\nKEYBOARD INTERRUPT")
    try:
        if episodes_processed > 0:
            save = input("Save model data? [y/n] ")
            if save == 'y' or save == 'Y':
                if run_as == Run.local:
                    env.close()
                    print(f"Saving model to {model_file}")
                    model.save(model_file)
                    if episodes_processed >= 3:
                        print(f"Saving reward plot to {figure_file}")
                        plot_reward(rewards, figure_file)
                elif run_as == Run.client:
                    env.close()
                    stub.SaveModel(mpm_pb2.SaveRequest(model_only=False))
                    stub.DropClient(mpm_pb2.Empty())
        elif run_as == Run.client:
            stub.DropClient(mpm_pb2.Empty())
        sys.exit(0)
    except SystemExit:
        os._exit(0)