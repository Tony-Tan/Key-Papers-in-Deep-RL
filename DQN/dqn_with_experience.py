import gym
import DQN.Network as Network
import cv2
import torch.utils.data as utils_data
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import numpy as np
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import random


class CustomDataset(utils_data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, item):
        return self.data[item], self.labels[item]

    def __len__(self):
        return len(self.data)


class Agent:
    def __init__(self, env, k_frames=4):
        self.env = env
        self.env_action_n = self.env.action_space.n
        self.state_value_function = Network.Net(4, self.env_action_n)
        self.k_frames = k_frames
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.state_value_function.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.state_value_function.parameters(), lr=0.001, momentum=0.9)
        self.writer = SummaryWriter("./log/")

    def convert_and_down_sampling(self, state):
        image = np.array(state)
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.resize(gray_img, (84, 110))
        return gray_img[26:110, 0:84]

    def select_action(self, state_phi, epsilon):
        policies = np.zeros(self.env.action_space.n)
        state_phi_tensor = torch.from_numpy(state_phi).unsqueeze(0).to(self.device)
        state_action_values = self.state_value_function(state_phi_tensor).cpu().detach().numpy()
        value_of_action_list = state_action_values[0]
        optimal_action = np.random.choice(
            np.flatnonzero(value_of_action_list == value_of_action_list.max()))
        for action_iter in range(self.env.action_space.n):
            if action_iter == optimal_action:
                policies[action_iter] = 1 - epsilon + epsilon / self.env.action_space.n
            else:
                policies[action_iter] = epsilon / self.env.action_space.n
        probability_distribution = policies
        action = np.random.choice(self.env.action_space.n, 1, p=probability_distribution)
        return action[0]

    def phi(self, episode):
        return (np.array([episode[-1], episode[-2], episode[-3], episode[-4]]) / 255. - 0.5).astype(np.float32)

    def skip_k_frame(self, action):
        new_state = 0
        reward = 0
        is_done = 0
        others = 0
        for i in range(self.k_frames):
            new_state, reward, is_done, others = self.env.step(action)
            if is_done:
                break
        return new_state, reward, is_done, others

    def train_state_value_function(self, epoch_num, memory, bach_size, gamma=0.9):
        total_loss = 0
        for epo_i in range(epoch_num):
            # build label:
            is_done_array = [mem[4] for mem in memory]
            next_state_data = [mem[3] for mem in memory]
            reward_array = [mem[2] for mem in memory]
            action_array = [mem[1] for mem in memory]
            state_data = [mem[0] for mem in memory]
            next_state_max_value = []
            if is_done_array[-1]:
                next_state_data = next_state_data[0:-1]
            fake_labels = np.zeros(len(next_state_data))
            next_state_data_set = CustomDataset(next_state_data, fake_labels)
            next_state_data_loader = DataLoader(next_state_data_set, batch_size=bach_size, shuffle=False)
            with torch.no_grad():
                for i, data in enumerate(next_state_data_loader, 0):
                    # get the inputs; data is a list of [inputs, labels]
                    inputs, labels = data
                    inputs = inputs.to(self.device)
                    outputs = self.state_value_function(inputs)
                    _, preds = torch.max(outputs, 1)
                    outputs = outputs.cpu().numpy()
                    preds = preds.cpu().numpy()
                    for pred_i in range(len(preds)):
                        next_state_max_value.append(outputs[pred_i][preds[pred_i]])

            if is_done_array[-1]:
                next_state_max_value.append(0)
            reward_array = np.array(reward_array) + gamma * np.array(next_state_max_value)

            labels_array = []
            for v_i in range(len(reward_array)):
                label = np.zeros(self.env_action_n)
                label[action_array[v_i]] = reward_array[v_i]
                labels_array.append(label.astype(np.float32))

            # train the model
            data_set = CustomDataset(state_data, labels_array)
            data_loader = DataLoader(data_set, batch_size=bach_size, shuffle=True)
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
                total_loss += loss.item()

    def running(self, episodes_num, mini_bach_size):
        memory = []
        for episode_i in range(episodes_num):
            total_reward = 0
            state_list = []
            state = self.env.reset()
            for i in range(0, 4):
                state_list.append(self.convert_and_down_sampling(state))
            state_phi = self.phi(state_list)
            action = self.select_action(state_phi, 0.5)
            new_state, reward, is_done, _ = self.skip_k_frame(action)
            total_reward += reward
            new_state = np.array(new_state)
            state_list.append(self.convert_and_down_sampling(new_state))
            new_state_phi = self.phi(state_list)
            memory.append([state_phi, action, reward, new_state_phi, is_done])

            while not is_done:
                state_phi = new_state_phi
                action = self.select_action(state_phi, 0.5)
                new_state, reward, is_done, _ = self.skip_k_frame(action)
                total_reward += reward
                new_state = np.array(new_state)
                state_list.append(self.convert_and_down_sampling(new_state))
                new_state_phi = self.phi(state_list)
                memory.append([state_phi, action, reward, new_state_phi, is_done])
                # skip the next k frames by executing the last action
                if len(memory) > mini_bach_size*4 and random.randint(0,100)<50:
                    sub_memory = random.sample(memory, mini_bach_size)
                    self.train_state_value_function(epoch_num=1, memory=sub_memory, bach_size=mini_bach_size)
            if len(memory) > 1024:
                memory=memory[len(memory)-1024:len(memory)]
            print("reward of episode: " + str(episode_i) + " is " + str(total_reward))
            self.writer.add_scalar('reward of episode', total_reward, episode_i)
        return


if __name__ == '__main__':
    env = gym.make('Breakout-v0')
    agent = Agent(env)
    agent.running(episodes_num=50000, mini_bach_size=64)
