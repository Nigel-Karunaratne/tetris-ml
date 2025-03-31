import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device is: {device}")

class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # Input layer
        self.fc2 = nn.Linear(128, 128)         # Hidden layer
        self.fc3 = nn.Linear(128, output_size) # Output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Activation for hidden layer 1
        x = F.relu(self.fc2(x))  # Activation for hidden layer 2
        x = self.fc3(x)          # Output layer (Q-values)
        return x

import random
import numpy as np
from collections import deque
from tetris import TetrisAction

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)  # Experience replay buffer
        self.gamma = 0.99                # Discount factor
        self.epsilon = 1.0              # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.device = device
        self.model = QNetwork(state_size, action_size).to(self.device)  # Q-network
        self.target_model = QNetwork(state_size, action_size).to(self.device)  # Target Q-network
        self.target_model.load_state_dict(self.model.state_dict())  # Initially copy model weights
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        # state = torch.tensor(state, dtype=torch.float32, device=self.device)
        # state = state.clone().detach()
        if np.random.rand() <= self.epsilon:
            return random.choice(list(TetrisAction))        # Random action for exploration
        
        if isinstance(state, torch.Tensor):
            state = state.clone().detach().unsqueeze(0)  # Detach from computation graph
        else:
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        # state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)       # Convert state to tensor
        q_values = self.model(state)
        return TetrisAction(torch.argmax(q_values).item())  # Return action with max Q-value
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in batch:
            if isinstance(state, torch.Tensor):
                state = state.clone().detach().unsqueeze(0)  # Detach from computation graph
            else:
                state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            if isinstance(next_state, torch.Tensor):
                next_state = next_state.clone().detach().unsqueeze(0)  # Detach from computation graph
            else:
                next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)            

            # state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            # next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
            target = reward
            if not done:
                next_q_values = self.target_model(next_state)
                target += self.gamma * torch.max(next_q_values)
            
            target_f = self.model(state)
            target_f[0][int(action)] = target  # Set the target Q-value for the chosen action
            
            self.optimizer.zero_grad()
            loss = F.mse_loss(target_f, self.model(state))
            loss.backward()
            self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay  # Decay epsilon
    
    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay  # Decay epsilon after each episode

from tetris import TetrisGame

def train_tetris():
    game = TetrisGame(gameSpeed=1000, humanGame=False, visual=True)
    agent = DQNAgent(state_size=game.get_state_size(), action_size=len(TetrisAction))  # Action size is the number of enum members
    episodes = 1000
    batch_size = 32

    print("Starting training... now!")
    for e in range(episodes):
        state = game.reset_game()  # Get initial state
        state = torch.FloatTensor(state).to(agent.device)
        isDone = False
        while not isDone:
            action = agent.act(state)  # Select an action using epsilon-greedy policy (returns an Action enum)
            reward, isDone, next_state = game.play_step(action)  # Apply the action (game.step expects an Action enum)
            agent.remember(state, action, reward, next_state, isDone)  # Store the experience in memory
            agent.replay(batch_size)  # Train the model on a batch of experiences
            state = next_state  # Move to the next state

        agent.decay_epsilon()  # Decay epsilon after each episode
        print(f"Episode {e}/{episodes} completed")

