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
import random
import time
import os


class CustomDataset(utils_data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, item):
        state_4_channels = np.array([self.data[item], self.data[item + 1],
                                     self.data[item + 2], self.data[item + 3]]).astype(np.float32)
        return state_4_channels, self.labels[item]

    def __len__(self):
        return len(self.data) - 3


class Agent:
    def __init__(self, env, k_frames=4, phi_temp_size=4, model_path='./model/'):
        self.env = env
        self.env_action_n = self.env.action_space.n
        self.state_value_function = Network.Net(4, self.env_action_n)
        self.k_frames = k_frames
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.state_value_function.to(self.device)
        self.criterion = nn.SmoothL1Loss() # nn.MSELoss()
        self.optimizer = optim.SGD(self.state_value_function.parameters(), lr=0.001, momentum=0.9)
        self.writer = SummaryWriter("./log/")
        self.down_sample_size = 84
        self.model_path = model_path
        self.phi_temp = []
        self.phi_temp_size = phi_temp_size
        if os.path.exists(self.model_path + 'last_model.txt'):
            file = open(self.model_path + 'last_model.txt', 'r')
            line = file.readlines()[0]
            file.close()
            print('found model file. \nloading model....')
            self.state_value_function.load_state_dict(torch.load(self.model_path + line))

    def convert_down_sample(self, state):
        image = np.array(state)
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.resize(gray_img, (self.down_sample_size, 110))
        return gray_img[110 - self.down_sample_size:110, 0:self.down_sample_size] / 255. - 0.5

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

    def phi(self):
        if len(self.phi_temp) > self.phi_temp_size:
            self.phi_temp = self.phi_temp[len(self.phi_temp) - self.phi_temp_size:len(self.phi_temp)]
        return (np.array(self.phi_temp)).astype(np.float32)

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

    def train_state_value_function(self, epoch_num, memory, bach_size, gamma=1.0):

        total_loss = 0
        for epo_i in range(epoch_num):
            # build label:
            is_done_array = [mem[4] for mem in memory]
            next_state_data = [mem[3] for mem in memory]
            reward_array = [mem[2] for mem in memory]
            action_array = [mem[1] for mem in memory]
            state_data = [mem[0] for mem in memory]

            next_state_max_value = []
            fake_labels = np.zeros(len(next_state_data))
            next_state_data_set = CustomDataset(next_state_data, fake_labels)
            next_state_data_loader = DataLoader(next_state_data_set, batch_size=bach_size, shuffle=False)

            with torch.no_grad():
                for i, data in enumerate(next_state_data_loader, 0):
                    # get the inputs; data is a list of [inputs, labels]
                    inputs, labels = data
                    inputs = inputs.to(self.device)
                    outputs = self.state_value_function(inputs)
                    _, predictions = torch.max(outputs, 1)
                    outputs = outputs.cpu().numpy()
                    predictions = predictions.cpu().numpy()
                    for p_i in range(len(predictions)):
                        next_state_max_value.append(outputs[p_i][predictions[p_i]])

            for is_done_i in range(len(is_done_array) - 3):
                if is_done_array[is_done_i + 3]:
                    next_state_max_value[is_done_i] = 0
            reward_array = np.array(reward_array)[3:] + gamma * np.array(next_state_max_value)

            labels_array = []
            reward_array = reward_array.astype(np.float32)
            for v_i in range(len(reward_array)):
                label = np.zeros(self.env_action_n)
                label[action_array[v_i + 3]] = reward_array[v_i]
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
                for label_i in range(len(outputs)):
                    for output_i in range(len(outputs[label_i])):
                        if labels[label_i][output_i] == 0:
                            outputs[label_i][output_i] = 0
                loss = self.criterion(outputs, labels)
                loss.backward()

                self.optimizer.step()
                total_loss += loss.item()

                if i % 100 == 0 and i != 0:
                    # record
                    iteration = (epo_i - 1) * len(data_loader) + i
                    self.writer.add_scalar('train/loss', total_loss / 100., iteration)
                    now = int(round(time.time() * 1000))
                    now02 = time.strftime('%H-%M-%S', time.localtime(now / 1000))
                    print(now02 + ' [%d, %d] loss: %.7f' %
                          (epo_i + 1, i + 1, total_loss / 100.))
                    total_loss = 0.0

    def running(self, episodes_num, mini_bach_size, memory_length=30000):
        memory = []
        frame_num = 0
        for episode_i in range(1, episodes_num):
            epsilon = 0.4
            total_reward = 0
            action = np.random.choice(self.env_action_n, 1)[0]
            state = self.env.reset()
            state = self.convert_down_sample(np.array(state))
            self.phi_temp.append(state)
            for i in range(1, 4):
                new_state, reward, is_done, _ = self.skip_k_frame(action)
                frame_num += 1
                new_state = self.convert_down_sample(np.array(new_state))
                self.phi_temp.append(new_state)
                memory.append([state, action, reward, new_state, is_done])
                state = new_state
            state_phi = self.phi()
            action = self.select_action(state_phi, epsilon)
            new_state, reward, is_done, _ = self.skip_k_frame(action)
            frame_num += 1
            total_reward += reward
            new_state = self.convert_down_sample(np.array(new_state))
            self.phi_temp.append(new_state)
            memory.append([state, action, reward, new_state, is_done])
            state = new_state
            while not is_done:
                state_phi = self.phi()
                action = self.select_action(state_phi, epsilon)
                new_state, reward, is_done, _ = self.skip_k_frame(action)
                frame_num += 1
                total_reward += reward
                new_state = self.convert_down_sample(np.array(new_state))
                memory.append([state, action, reward, new_state, is_done])
                self.phi_temp.append(new_state)
                state = new_state
                # skip the next k frames by executing the last action
            if len(memory) > memory_length:
                self.train_state_value_function(epoch_num=1, memory=memory, bach_size=mini_bach_size)
                memory = memory[len(memory) - memory_length:len(memory)]
            print("reward of episode: " + str(episode_i) + " is " + str(total_reward)
                  + " and frame number is " + str(frame_num))
            self.writer.add_scalar('reward of episode', total_reward, episode_i)
            if episode_i % 500 == 0:
                print('model saved')
                now = int(round(time.time() * 1000))
                now02 = time.strftime('%m-%d-%H-%M-%S', time.localtime(now / 1000))
                torch.save(self.state_value_function.state_dict(), self.model_path + now02 + '.pth')
                file = open(self.model_path + 'last_model.txt', 'w+')
                file.write(now02 + '.pth')
                file.close()
        return


if __name__ == '__main__':
    env = gym.make('Pong-v0')
    agent = Agent(env)
    agent.running(episodes_num=100000, mini_bach_size=32)
