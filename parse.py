import sys
from enum import Enum

#/ INITALIZE PARAMETERS AND PARSE FLAGS /#

Run = Enum("Run", "local server client")

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