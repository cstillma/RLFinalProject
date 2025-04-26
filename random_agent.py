import gymnasium as gym
import numpy as np

# Define a random agent for use as a comparison baseline for the DQN
class RandomAgent:
    # Initialize the agent with the environment's action space
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation):
        # Select a random action from the action space
        return self.action_space.sample()

# Function to set up and run the random agent    
def RA_main():
    env = gym.make("CarRacing-v3", continuous = False) # import the car racing environment and use the discrete action space
    agent = RandomAgent(env.action_space) # initialize the random agent
    for episode in range(5): # run five episodes
        observation, info = env.reset() # reset the environment after each episode to get the observation and other info
        done = False
        episode_reward = 0 # initialize episode reward
        while not done: # loop until the episode is complete
            action = agent.act(observation) # randomly select an action
            observation, reward, done, truncated, info = env.step(action) # apply the action to the environment, receive the next state and reward
            episode_reward += reward # add the reward for this step
            # env.render() # visualize the environment (must add render_mode = "human" to gym.make)
        print(f"Episode {episode + 1} finished with a score of reward: {episode_reward}")
    env.close() # close the environment

if __name__ == "__main__":
    RA_main()