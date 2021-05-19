import os
import numpy as np
import matplotlib.pyplot as plt
from keyboard import is_pressed

#/ PROCESS ACTIONS AND SAVE FIGURE /#

episodes_processed = 0
rewards = []

# Determine if action should be random or from input, handle prediction elsewhere
def get_action(always_noop, epsilon, sample):
    action = -1
    action_type = ''

    if is_pressed("space"):
        if always_noop:
            print("—————————————————————————————————————————————————————")
            print("                 DISABLED ALWAYS NOOP                ")
            print("—————————————————————————————————————————————————————")
            always_noop = False
        else:
            print("—————————————————————————————————————————————————————")
            print("                 ENABLED ALWAYS NOOP                 ")
            print("—————————————————————————————————————————————————————")
            always_noop = True
    elif is_pressed("up") and is_pressed("right"):
        action = 5
    elif is_pressed("up") and is_pressed("left"):
        action = 6
    elif is_pressed("down") and is_pressed("right"):
        action = 7
    elif is_pressed("down") and is_pressed("left"):
        action = 8
    elif is_pressed("up"):
        action = 1
    elif is_pressed("right"):
        action = 2
    elif is_pressed("left"):
        action = 3
    elif is_pressed("down"):
        action = 4

    if action != -1:
        action_type = "(inpt)"
    else:
        # Process predicted or random environment actions
        rand = np.random.random()
        if always_noop:
            action = 0
        elif rand < epsilon:
            action = sample
            action_type = "(rand)"

    return action, action_type, always_noop

# Plot total reward over each episode
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
