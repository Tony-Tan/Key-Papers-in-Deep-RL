import gym
import DQN.Network as Network
import cv2
import torch.utils.data as uData
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import numpy as np
import torch.nn as nn


class CustomDataset(uData.Dataset):
    def __init__(self, memory, action_space_n):
        self.data = [mem[0] for mem in memory]
        self.labels = []
        for mem in memory:
            label = np.zeros(action_space_n)
            label[mem[1]] = mem[2]
            self.labels.append(label.astype(np.float32))

    def __getitem__(self, item):
        return self.data[item], self.labels[item]

    def __len__(self):
        return len(self.data)


class Agent:
    def __init__(self, env, k_frames=4):
        self.env = env
        self.env_action_n = self.env.action_space.n
        self.state_value_function = Network.Net(4,self.env_action_n)
        self.k_frames = k_frames
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.state_value_function.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.state_value_function.parameters(), lr=0.001, momentum=0.9)

    def convert_and_down_sampling(self, state):
        image = np.array(state)
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.resize(gray_img, (84, 110))
        return gray_img[26:110, 0:84]

    def select_action(self, state_phi):
        state_action_values = np.ones(self.env_action_n) / self.env_action_n
        state_action_values = np.array(state_action_values)
        state_action_values /= np.sum(state_action_values)
        action_selected = np.random.choice(self.env_action_n, 1, p=state_action_values)
        return action_selected[0]

    def phi(self, episode):
        return (np.array([episode[-1], episode[-2], episode[-3], episode[-4]])/255.).astype(np.float32)

    def skip_k_frame(self, action):
        for i in range(self.k_frames):
            self.env.step(action)

    def running(self, repeat_times, mini_bach_size):
        for repeat_i in range(repeat_times):
            memory = []
            state_list = []
            state = self.env.reset()
            while len(state_list) < 3:
                state_list.append(self.convert_and_down_sampling(state))
            state_list.append(self.convert_and_down_sampling(state))
            state_phi = self.phi(state_list)
            action = self.select_action(state_phi)
            new_state, reward, is_done, _ = self.env.step(action)
            new_state = np.array(new_state)
            state_list.append(self.convert_and_down_sampling(new_state))
            new_state_phi = self.phi(state_list)
            memory.append([state_phi, action, reward, new_state_phi])
            # skip the next k frames by executing the last action
            self.skip_k_frame(action)
            while not is_done:
                state_phi = new_state_phi
                action = self.select_action(state_phi)
                new_state, reward, is_done, _ = self.env.step(action)
                new_state = np.array(new_state)
                state_list.append(self.convert_and_down_sampling(new_state))
                new_state_phi = self.phi(state_list)
                memory.append([state_phi, action, reward, new_state_phi])
                # skip the next k frames by executing the last action
                self.skip_k_frame(action)
                if len(memory) > mini_bach_size:
                    data_set = CustomDataset(memory, self.env_action_n)
                    data_loader = DataLoader(data_set, batch_size=mini_bach_size, shuffle=True)
                    for i, data in enumerate(data_loader, 0):
                        # get the inputs; data is a list of [inputs, labels]
                        inputs, labels = data
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                        # zero the parameter gradients
                        self.optimizer.zero_grad()
                        # forward + backward + optimize
                        outputs = self.state_value_function(inputs)
                        loss = self.criterion(outputs, labels)
                        loss.backward()
                        self.optimizer.step()
        return


if __name__ == '__main__':
    env = gym.make('Breakout-v0')
    agent = Agent(env)
    agent.running(repeat_times=1, mini_bach_size=10)
