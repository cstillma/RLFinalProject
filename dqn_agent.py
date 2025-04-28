import numpy as np
import random
import tensorflow as tf
from collections import deque

class DQN_Agent:
    # Initialize the DQN agent and hyperparameters
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen = 2000) # replay buffer stores only the 2000 most recent experiences for training
        self.gamma = 0.95 # discount factor - determines how much future rewards are valued compared to immediate rewards
        self.epsilon = 1.0 # exploration rate
        self.epsilon_min = 0.01 # minimum exploration rate
        self.epsilon_decay = 0.995 # decay rate to shift agent from exploration to exploitation
        self.learning_rate = 0.001 # learning rate - step size used by optimizer to update neural network weights
        self.model = self._build_model() # neural network 

    # Build the neural network for Q-value approximation using TensorFlow/Keras
    def _build_model(self):
        inputs = tf.keras.Input(shape=(27648, )) # flattened input shape: 96 * 96 * 3
        x = tf.keras.layers.Dense(24, activation = 'relu')(inputs) # first hidden layer
        x = tf.keras.layers.Dense(24, activation = 'relu')(x) # second hidden layer
        outputs = tf.keras.layers.Dense(self.action_size, activation = 'linear')(x) # output layer

        # Build the model
        model = tf.keras.Model(inputs = inputs, outputs = outputs)
        
        # model = tf.keras.Sequential([
        #     tf.keras.layers.Dense(24, input_dim = self.state_size, activation = 'relu'), # input layer takes in current state (state_size dimensions). 
        #     tf.keras.layers.Dense(24, activation = 'relu'), # hidden layer adds complexity to the network by learning higher order features from input state
        #     tf.keras.layers.Dense(self.action_size, activation = 'linear') # output layer outputs Q-values for each possible action (action_size dimensions)
        #     ])

        # MSE loss function measures difference between predicted Q-values and target Q-values
        # Adam optimizer updates weights using gradient descent
        model.compile(loss = 'mse', optimizer = tf.keras.optimizers.Adam(learning_rate = self.learning_rate)) 
        return model
    
    # Store experiences in the replay buffer for later training
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # append each new experience to the memory queue

    # Choose next action using epsilon-greedy policy
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size) # explore: select a random action
        state = np.array(state).flatten().reshape(1, -1) # ensure the state is the correct shape for the model's input dimensions
        q_values = self.model.predict(state, verbose = 0) # predict Q-values
        return np.argmax(q_values[0]) # exploit: action with max Q-value
    
    # Train the neural network using experiences in the replay buffer
    def replay(self):
        if len(self.memory) < 32: # skip training if not enough experience
            return
        minibatch = random.sample(self.memory, 32) # randomly sample 32 previous experiences

        for state, action, reward, next_state, done in minibatch:
            # Reshape the state and next_state to match model's input requirements
            state = np.array(state).flatten().reshape(1, -1) # flatten and reshape the state
            next_state = np.array(next_state).flatten().reshape(1, -1) # flatten and reshape the next state
            # Compute target Q-values
            target = reward # initialize target to the immediate reward
            if not done: # if the episode is not done, the target is updated with future rewards
                target += self.gamma * np.amax(self.model.predict(next_state, verbose = 0)[0]) # incorporates the discounted maximum Q-value from the next state to encourage the agent to maximize long-term rewards
            # Fit the model
            target_f = self.model.predict(state, verbose = 0) # predict Q-values for the current state
            target_f[0][action] = target # update the target Q-value for the chosen action
            self.model.fit(state, target_f, epochs = 1, verbose = 0) # model is trained using the updated Q-values
        
        # Decay epsilon after every training step to shift agent from exploration to exploitation
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

