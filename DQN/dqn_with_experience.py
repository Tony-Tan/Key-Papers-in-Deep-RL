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
from collections import deque
import random
import time
import os


class CustomDataset(utils_data.Dataset):
    def __init__(self, data, labels):
        self.data = np.array(data)
        self.labels = labels

    def __getitem__(self, item):
        return self.data[item], self.labels[item]

    def __len__(self):
        return len(self.data)


class Agent:
    def __init__(self, env, k_frames=1, phi_temp_size=4, model_path='./model/'):
        self.env = env
        self.env_action_n = self.env.action_space.n
        self.state_value_function = Network.Net(4, self.env_action_n)
        self.k_frames = k_frames
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.state_value_function.to(self.device)
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
        gray_img = cv2.resize(gray_img, (92, 110))
        return gray_img[110 - 92:110, 0:92] / 255 - 0.5

    def select_action(self, state_phi, epsilon):
        policies = np.zeros(self.env.action_space.n)
        state_phi_tensor = torch.from_numpy(state_phi).unsqueeze(0).to(self.device)
        state_action_values = self.state_value_function(state_phi_tensor).cpu().detach().numpy()
        value_of_action_list = state_action_values[0]
        optimal_action = np.random.choice(
            np.flatnonzero(value_of_action_list == value_of_action_list.max()))
        for action_iter in range(self.env.action_space.n):
            if action_iter == optimal_action:
                policies[action_iter] = 1. - epsilon + epsilon / self.env.action_space.n
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
            new_state, r, is_done, others = self.env.step(action)
            reward += r
            if is_done:
                break
        return new_state, reward, is_done, others

    def train_state_value_function(self, memory, bach_size, gamma=0.999):

        # training parameter

        criterion = nn.SmoothL1Loss()
        optimizer = optim.Adam(self.state_value_function.parameters(), lr=1e-6)
        total_loss = 0
        # build label:
        next_state_data = [mem[3] for mem in memory]
        reward_array = [mem[2] for mem in memory]
        action_array = [[int(mem[1])] for mem in memory]
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
                # if random.randint(0, 10000) < 10:
                #     print(predictions)
                #     print(next_state_max_value)
        for is_done_i in range(len(memory)):
            if memory[is_done_i][4]:
                next_state_max_value[is_done_i] = 0
        reward_array = np.array(reward_array) + gamma * np.array(next_state_max_value)
        reward_array = reward_array.transpose().astype(np.float32)
        action_array = torch.Tensor(action_array).long()
        # labels_array = []
        # reward_array = reward_array.astype(np.float32)
        # for v_i in range(len(reward_array)):
        #     label = np.zeros(self.env_action_n)
        #     label[action_array[v_i]] = reward_array[v_i]
        #     labels_array.append(label.astype(np.float32))

        # train the model
        # data_set = CustomDataset(state_data, labels_array)
        data_set = CustomDataset(state_data, reward_array)
        data_loader = DataLoader(data_set, batch_size=bach_size, shuffle=False)
        for i, data in enumerate(data_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(self.device)
            labels = labels.to(self.device).view(-1, 1)
            actions = action_array.to(self.device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = self.state_value_function(inputs).gather(0, actions)

            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()
            total_loss += loss.item()
            # record
            self.writer.add_scalar('train/loss', total_loss)
            # now = int(round(time.time() * 1000))
            # now02 = time.strftime('%H-%M-%S', time.localtime(now / 1000))
            # print(now02 + ' [%d, %d] loss: %.7f' %
            #       (epo_i + 1, i + 1, total_loss / 100.))
            # total_loss = 0.0

    def running(self, episodes_num, mini_bach_size,epsilon_start=0.9,
                epsilon_end=0.1, epsilon_update=100, memory_length=20000):
        memory = deque(maxlen=memory_length)
        frame_num = 0
        epsilon = epsilon_start
        for episode_i in range(1, episodes_num):
            # set a dynamic epsilon
            if episode_i % epsilon_update == 0 and epsilon > epsilon_end:
                epsilon -= 0.001
            total_reward = 0
            # random choice an action at the beginning of the process
            action = np.random.choice(self.env_action_n, 1)[0]
            state = self.env.reset()
            state = self.convert_down_sample(np.array(state))
            # add state into a list to create a phi
            # another three states are needed
            self.phi_temp.append(state)
            for i in range(1, 4):
                new_state, reward, is_done, _ = self.skip_k_frame(action)
                frame_num += 1
                new_state = self.convert_down_sample(np.array(new_state))
                self.phi_temp.append(new_state)
            # create phi
            state_phi = self.phi()
            # select action according the first phi
            action = self.select_action(state_phi, epsilon)
            new_state, reward, is_done, _ = self.skip_k_frame(action)
            frame_num += 1
            total_reward += reward
            # down sample and add the new state into the list to create phi
            new_state = self.convert_down_sample(np.array(new_state))
            self.phi_temp.append(new_state)
            new_state_phi = self.phi()
            # add phi(state) and phi(new state) and reward and action into memory
            memory.append([state_phi, action, reward, new_state_phi, is_done])

            while not is_done:
                state_phi = new_state_phi
                action = self.select_action(state_phi, epsilon)
                new_state, reward, is_done, _ = self.skip_k_frame(action)

                frame_num += 1
                total_reward += reward
                new_state = self.convert_down_sample(np.array(new_state))
                cv2.imshow('new_state', np.array((new_state + 0.5) * 255.).astype(np.uint8))
                cv2.waitKey(1)
                self.phi_temp.append(new_state)
                new_state_phi = self.phi()
                memory.append([state_phi, action, reward, new_state_phi, is_done])
                if len(memory) >= memory_length:
                    sub_memory = random.sample(memory, mini_bach_size)
                    self.train_state_value_function(memory=sub_memory, bach_size=mini_bach_size)
            print("reward of episode: " + str(episode_i) + " is " + str(total_reward)
                  + " and frame number is " + str(frame_num))
            self.writer.add_scalar('reward of episode', total_reward, episode_i)
            if episode_i % 100 == 0:
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
    agent.running(episodes_num=10000, mini_bach_size=32)
