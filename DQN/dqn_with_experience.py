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
import torch.nn.functional as F


class AgentDQN:
    def __init__(self, environment, mini_batch_size=32, episodes_num=100000, k_frames=4, input_frame_size=84, memory_length=2e4,
                 phi_temp_size=4, model_path='./model/', log_path='./log/'):
        # basic configuration
        self.__env = environment
        self.__action_n = environment.action_space.n
        self.__mini_batch_size = mini_batch_size
        self.__episodes_num = episodes_num
        self.__k_frames = k_frames
        self.__input_frame_size = input_frame_size
        self.__phi_temp = deque(maxlen=phi_temp_size)
        self.__phi_temp_size = phi_temp_size
        self.__memory = deque(maxlen=int(memory_length))
        # networks training configuration
        self.state_action_value_function = Network.Net(4, self.__action_n)
        self.__device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.__criterion = nn.SmoothL1Loss()
        self.__optimizer = optim.Adam(self.state_action_value_function.parameters(), lr=1e-6)
        self.state_action_value_function.to(self.__device)
        self.__writer = SummaryWriter(log_path)
        self.__model_path = model_path
        self.load_existing_model()

    def load_existing_model(self):
        if os.path.exists(self.__model_path + 'last_model.txt'):
            file = open(self.__model_path + 'last_model.txt', 'r')
            line = file.readlines()[0]
            file.close()
            print('found model file. \nloading model....')
            self.state_action_value_function.load_state_dict(torch.load(self.__model_path + line))

    def pre_process_and_add_state_into_phi_temp(self, state):
        """
        :param state: 2-d int matrix, original state of environment
        :return: 2-d float matrix, 1-channel image with size of self.down_sample_size
                 and the value is converted to [-0.5,0.5]
        """
        image = np.array(state)
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.resize(gray_img, (self.__input_frame_size, 100))
        gray_img = gray_img[100 - self.__input_frame_size:100, 0:self.__input_frame_size]/255. - 0.5
        self.__phi_temp.append(gray_img)

    def select_action(self, state_phi, epsilon):
        """
        :param state_phi:  the last 4 states of the environment after convert and down sample
        :param epsilon: float, exploitation or exploration by epsilon-greedy strategy
        :return: int, an action
        """
        state_phi_tensor = torch.from_numpy(state_phi).unsqueeze(0).to(self.__device)
        state_action_values = self.state_action_value_function(state_phi_tensor).cpu().detach().numpy()
        value_of_action_list = state_action_values[0]
        optimal_action = np.random.choice(
            np.flatnonzero(value_of_action_list == value_of_action_list.max()))
        if random.randint(0, 1000) < epsilon * 1000.:
            return random.randint(0, self.__action_n - 1)
        else:
            return optimal_action

    def phi(self):
        """
        :return: self.phi_temp_size x state_size x state_size tensor, the last self.phi_temp_size
        """
        if len(self.__phi_temp) == self.__phi_temp_size:
            return (np.array(self.__phi_temp)).astype(np.float32)

    def skip_k_frame(self, action):
        """
        :param action: an int number, taking the same action in the following self.k_frames frames
        :return: the last state, the summation of reward, is_done, others. If the episode stop, the function stop
        """
        new_state = 0
        reward = 0
        is_done = 0
        others = 0
        for i in range(self.__k_frames):
            new_state, r, is_done, others = self.__env.step(action)
            reward += r
            if is_done:
                break
        return new_state, reward, is_done, others

    def train_network(self, memory, gamma=0.99):
        """
        training the network of state value function
        :param memory: memory of states
        :param gamma: float number, decay coefficient
        :return: nothing
        """
        # training parameter
        total_loss = 0
        # build label:
        next_state_data = np.array([mem[3] for mem in memory])
        reward_array = np.array([mem[2] for mem in memory])
        action_array = np.array([[int(mem[1])] for mem in memory])
        state_data = np.array([mem[0] for mem in memory])
        next_state_max_value = []
        with torch.no_grad():
            inputs = torch.from_numpy(next_state_data).to(self.__device)
            outputs = self.state_action_value_function(inputs)
            _, predictions = torch.max(outputs, 1)
            outputs = outputs.cpu().numpy()
            predictions = predictions.cpu().numpy()
            for p_i in range(len(predictions)):
                next_state_max_value.append(outputs[p_i][predictions[p_i]])
        for is_done_i in range(len(memory)):
            if memory[is_done_i][4]:
                next_state_max_value[is_done_i] = 0
        reward_array = reward_array + gamma * np.array(next_state_max_value)
        reward_array = reward_array.transpose().astype(np.float32)
        action_array = torch.Tensor(action_array).long()
        # train the model
        inputs = torch.from_numpy(state_data).to(self.__device)
        labels = torch.from_numpy(reward_array).to(self.__device).view(-1, 1)
        actions = action_array.to(self.__device)
        # zero the parameter gradients
        self.__optimizer.zero_grad()
        # forward + backward + optimize
        outputs = self.state_action_value_function(inputs).gather(1, actions)
        loss = F.mse_loss(outputs, labels)
        # Minimize the loss
        self.__optimizer.zero_grad()
        loss.backward()
        self.__optimizer.step()
        total_loss += loss.item()
        # record
        self.__writer.add_scalar('train/loss', total_loss)

    def learning_an_episode(self,epsilon):
        frame_num = 0
        total_reward = 0
        state = self.__env.reset()
        self.pre_process_and_add_state_into_phi_temp(state)
        is_done = False
        for i in range(1, self.__phi_temp_size):
            action = np.random.choice(self.__action_n, 1)[0]
            new_state, reward, is_done, _ = self.skip_k_frame(action)
            self.pre_process_and_add_state_into_phi_temp(new_state)
            if is_done:
                return is_done
        state_phi = self.phi()
        while not is_done:
            action = self.select_action(state_phi, epsilon)
            new_state, reward, is_done, _ = self.skip_k_frame(action)
            frame_num += 1
            total_reward += reward
            self.pre_process_and_add_state_into_phi_temp(new_state)
            new_state_phi = self.phi()
            self.__memory.append([state_phi, action, reward, new_state_phi, is_done])
            if len(self.__memory) > self.__mini_batch_size:
                sub_memory = random.sample(self.__memory, self.__mini_batch_size)
                self.train_network(memory=sub_memory)
            state_phi = new_state_phi
        return frame_num, total_reward

    def save_model(self):
        # save model files
        print('model saved')
        now = int(round(time.time() * 1000))
        now02 = time.strftime('%m-%d-%H-%M-%S', time.localtime(now / 1000))
        torch.save(self.state_action_value_function.state_dict(), self.__model_path + now02 + '.pth')
        file = open(self.__model_path + 'last_model.txt', 'w+')
        file.write(now02 + '.pth')
        file.close()

    def record_reward(self, frame_num, total_reward, epsilon, episode_i):
        # print and record reward and loss
        print("reward of episode: " + str(episode_i) + " is " + str(total_reward)
              + " and frame number is " + str(frame_num) + ' epsilon: ' + str(epsilon))
        self.__writer.add_scalar('reward of episode', total_reward, episode_i)

    def learning(self, episodes_num, epsilon_max=1.0,
                 epsilon_min=0.1, epsilon_decay=0.99995):
        """
        :param episodes_num: int number, how many episodes would be run
        :param epsilon_max: float number, epsilon start number, 1.0 for most time
        :param epsilon_min: float number, epsilon end number, 0.1 in the paper
        :param epsilon_decay: float number, decay coefficient of epsilon
        :return: nothing
        """
        frame_num = self.__phi_temp_size
        epsilon = epsilon_max
        for episode_i in range(1, self.__episodes_num):
            # set a dynamic epsilon
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            # random choice an action at the beginning of the process
            frame_num_i, reward_i = self.learning_an_episode(epsilon)
            frame_num += frame_num_i
            self.record_reward(frame_num, reward_i, epsilon, episode_i)
            if episode_i % 500 == 0:
                self.save_model()


if __name__ == '__main__':
    env = gym.make('Pong-v0')
    agent = AgentDQN(env)
    agent.learning()
