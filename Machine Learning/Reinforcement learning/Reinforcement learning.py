# Load OpenAI Gym and other necessary packages
import gym
import random
import numpy
import time

# Environment
env = gym.make("Taxi-v3")

# Training parameters for Q learning
alpha = 0.9 # Learning rate
gamma = 0.9 # Future reward discount factor
epsilon = 0.2 # Try random new thing to find better solution
num_of_episodes = 1000
num_of_steps = 500 # per each episode

# Q tables for rewards
Q_reward = numpy.zeros((500,6)) # All same
#Q_reward = -100000*numpy.random.random((500, 6)) # Random

# Training w/ random sampling of actions
# YOU WRITE YOUR CODE HERE
for i in range(num_of_episodes):
    state = env.reset()

    reward = 0, 0
    done = False # optional run till done, not follow num_of_steps
    # while not done:
    for n in range(num_of_steps):
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Do new thing
        else:
            action = numpy.argmax(Q_reward[state])  # Do base on table

        next_state, reward, done, info = env.step(action)
        old_value = Q_reward[state, action]
        next_max = numpy.max(Q_reward[next_state])
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        Q_reward[state, action] = new_value
        state = next_state

# Testing
repeat = 10
avg_reward = 0
avg_step = 0
for i in range(repeat):
    state = env.reset()
    tot_reward = 0
    tot_step = 0
    for t in range(50):
        action = numpy.argmax(Q_reward[state,:])
        state, reward, done, info = env.step(action)
        tot_reward += reward
        tot_step += 1
        env.render()
        time.sleep(1)
        if done:
            print("Total reward %d" %tot_reward)
            break
    avg_reward += tot_reward
    avg_step += tot_step

avg_reward = avg_reward/repeat
avg_step = avg_step/repeat
print(f"For {repeat} times:"
      f" Average reward: {avg_reward} and Average step: {avg_step}.")