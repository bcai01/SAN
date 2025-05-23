import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import torch.nn as nn
from copy import deepcopy
import random
from tqdm import tqdm
import math

import torch.nn as nn
import torch.nn.functional as F

epsilon = 1e-8

class PolicyNet(nn.Module):
    def __init__(self, num_teacher, embedding_length,device):
        super(PolicyNet, self).__init__()
        self.num_teacher = num_teacher  # 假设有两个老师模型

        # 定义权重和偏置
        self.W1 = nn.Parameter(torch.FloatTensor(embedding_length, 128).uniform_(-0.5, 0.5)) #768,128
        self.W2 = nn.Parameter(torch.FloatTensor(128, 128).uniform_(-0.5, 0.5)) #128,128
        self.W3 = nn.Parameter(torch.FloatTensor(num_teacher, 128).uniform_(-0.5, 0.5)) #2,128
        self.b = nn.Parameter(torch.FloatTensor(1, 128).uniform_(-0.5, 0.5))

        self.fc_alpha = nn.Parameter(torch.FloatTensor(128, 1).uniform_(-0.5, 0.5))  # 输出 alpha
        self.fc_beta = nn.Parameter(torch.FloatTensor(128, 1).uniform_(-0.5, 0.5))  # 输出 beta


        self.epsilon = 1.0  # 初始epsilon值
        self.epsilon_min = 0.01  # epsilon的最小值
        self.epsilon_decay = 0.995  # epsilon衰减率
        self.device = device
    def forward(self, x1, x2, x3):
        # x1: (num_sent, batch_size, embedding_length) 2,128,768
        # x2: (num_teacher, batch_size, batch_size) 2,128,128
        # x3: (num_teacher, 1) 1,2

        # 将输入与相应的权重相乘
        x1_ = torch.matmul(x1, self.W1)  #2,128,128
        x2_ = torch.matmul(x2, self.W2)  #2,128,128

        # 将第三个输入乘以其权重
        x3_ = torch.matmul(x3, self.W3)  #1,128

        # 将所有结果相加并加上偏置
        scaled_out = torch.relu(x1_ + x2_ + x3_ + self.b)
        # scaled_out = torch.clamp(scaled_out, min=1e-5, max=10 - 1e-5) #2,128,128
        # 重塑scaled_out以匹配全连接层的期望输入维度
        scaled_out_reshaped = scaled_out.view(-1, 128)  # 形状现在是 [-1, 128]
        # 添加全连接层以生成形状为 (batch_size*num_teacher, num_teacher) 的输出
        # 输出两个参数 alpha 和 beta
        alpha = torch.matmul(scaled_out_reshaped, self.fc_alpha)
        beta = torch.matmul(scaled_out_reshaped, self.fc_beta)
        alpha = F.softplus(alpha).mean() + 50
        #alpha = torch.relu(alpha).mean() + 50
        beta = F.softplus(beta).mean() + 50
        #beta = torch.relu(beta).mean() + 50
        weights = [alpha, beta]

        return weights

    def take_action(self, state):
        # 直接调用网络来生成一个概率
        weights = self.forward(*state)

        # 计算概率分布
        # dist = torch.distributions.Normal(weights[0].float(), weights[1].float())
        dist = torch.distributions.beta.Beta(weights[0].float(), weights[1].float())
        action = dist.sample().to(self.device)

        # 更新epsilon值，但不让它低于最小值
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        return action, weights

    def test_policy(self, state):
        avg_probability = self.forward(*state).to(self.device) # 获取动作概率
        action = torch.distributions.Bernoulli(avg_probability).sample().to(self.device)  # 选择概率最高的动作
        return action, avg_probability


from collections import namedtuple
import random
import torch
from torch.distributions import Bernoulli

# 假设你有一个表示概率的张量
probs = torch.tensor([0.5], requires_grad=True)  # 示例概

Transition = namedtuple('Transion',
                        ('state', 'action', 'weights', 'reward', 'value'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self):
        return self.memory

    def __len__(self):
        return len(self.memory)

    def clear(self):
        self.memory = []  # 清空memory
        self.position = 0  # 重置position为0


def optimize_model(memory, policy_net,critic, device, lr=1e-4):
    #设置一些超参数
    CLIP_EPSILON = 0.2
    NUM_PPO_UPDATES = 3  # 选择适当的更新次数
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    BATCH_SIZE = 10
    # 确保内存中有数据
    gamma = 0.99
    gae_lambda = 0.95
    num_batches = len(memory) // BATCH_SIZE
    all_transitions = memory.sample()
    batch = Transition(*zip(*all_transitions))
    for _ in range(NUM_PPO_UPDATES):

        # 准备数据
        # Prepare data
        action_batch = torch.cat(list(map(lambda a: torch.tensor([a], device=device), batch.action)))
        reward_batch = torch.cat(list(map(lambda r: torch.tensor([r], device=device), batch.reward)))
        old_weights = torch.cat(list(map(lambda r: torch.tensor(r, device=device), batch.weights)))
        old_weights = old_weights.view(-1, 2)
        # 将 batch.value 直接转换为张量
        value = torch.cat([torch.tensor([v], device=device) for v in batch.value])

        # 计算回报
        advantage = torch.zeros(len(reward_batch), dtype=torch.float32, device=device)
        for t in range(len(reward_batch) - 1):  # 逆序时序差分值
            discount = 1
            a_t = 0
            for k in range(t, len(reward_batch) - 1):
                a_t += 0.99 * (reward_batch[k] + gamma * value[k + 1] * - value[k])
            advantage[t] = a_t

        weights_list = []
        for state in batch.state:
            if isinstance(state, torch.Tensor):
                state = state.half().to(device)  # 转换为半精度并移至正确的设备
            elif isinstance(state, list):
                state = [s.float().to(device) for s in state]
            weight = policy_net(*state)
            weights1 = torch.stack([weight[0], weight[1]]).unsqueeze(0)
            weights_list.append(weights1)
        weights = torch.cat(weights_list, dim=0)




        # 计算对数概率和概率比率
        # m = torch.distributions.Normal(weights[:, 0].float(), weights[:, 1].float())
        m = torch.distributions.Beta(weights[:, 0].float(), weights[:, 1].float())
        log_probs = m.log_prob(action_batch)
        # beta_distribution = torch.distributions.Normal(old_weights[:, 0].float(), old_weights[:, 1].float())
        beta_distribution = torch.distributions.Beta(old_weights[:, 0].float(), old_weights[:, 1].float())
        old_log_probs = beta_distribution.log_prob(action_batch)
        ratio = torch.exp(log_probs - old_log_probs)

        # 计算 clipped ratio 和 PPO 目标函数
        clip_ratio = torch.clamp(ratio, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON)
        surrogate1 = ratio * advantage
        surrogate2 = clip_ratio * advantage
        ppo_loss = -torch.min(surrogate1, surrogate2).mean()

        value_list = []
        for state in batch.state:
            if isinstance(state, torch.Tensor):
                state = state.half().to(device)  # 转换为半精度并移至正确的设备
            elif isinstance(state, list):
                state = [s.float().to(device) for s in state]
            critic_value = critic(*state).unsqueeze(0)
            value_list.append(critic_value)
        values = torch.cat(value_list, dim=0)

        returns = advantage + value
        critic_loss = (returns - values) ** 2
        critic_loss = critic_loss.mean()
        total_loss = ppo_loss + 0.5 * critic_loss

        # 优化模型
        optimizer.zero_grad()
        critic.optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        critic.optimizer.step()



class actor(nn.Module):
    def __init__(self, policyNet, tau):
        super(actor, self).__init__()
        self.target_policy = policyNet
        self.active_policy = policyNet

    def get_target_logOutput(self, x1, x2, x3):
        out = self.target_policy(x1, x2, x3)
        logOut = torch.log(out)
        return logOut

    def get_target_output(self, x1, x2, x3, scope):
        if scope == "target":
            out = self.target_policy(x1, x2, x3)
        if scope == "active":
            out = self.active_policy(x1, x2, x3)
        return out

    def get_gradient(self, x1, x2, x3, reward, scope):
        if scope == "target":
            out = self.target_policy(x1, x2, x3)
            logout = torch.log(out).view(-1)
            index = reward.index(0)
            index = (index + 1) % 2
            # print(out, reward, index, logout[index].view(-1), logout)
            # print(logout[index].view(-1))
            grad = torch.autograd.grad(logout[index].view(-1),
                                       self.target_policy.parameters())  # torch.cuda.FloatTensor(reward[index])
            # print(grad[0].size(), grad[1].size(), grad[2].size())
            # print(grad[0], grad[1], grad[2])
            grad[0].data = grad[0].data * reward[index]
            grad[1].data = grad[1].data * reward[index]
            grad[2].data = grad[2].data * reward[index]
            # print(grad[0], grad[1], grad[2])
            return grad
        if scope == "active":
            out = self.active_policy(x1, x2, x3)
        return out

    def assign_active_network_gradients(self, grad1, grad2, grad3):
        params = [grad1, grad2, grad3]
        i = 0
        for name, x in self.active_policy.named_parameters():
            x.grad = deepcopy(params[i])
            i += 1

    def update_target_network(self):
        params = []
        for name, x in self.active_policy.named_parameters():
            params.append(x)
        i = 0
        for name, x in self.target_policy.named_parameters():
            x.data = deepcopy(params[i].data * (tau) + x.data * (1 - tau))
            i += 1

    def assign_active_network(self):
        params = []
        for name, x in self.target_policy.named_parameters():
            params.append(x)
        i = 0
        for name, x in self.active_policy.named_parameters():
            x.data = deepcopy(params[i].data)
            i += 1


import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(self, input_dim, num_teacher, embedding_length, hidden_dim=[256, 128], output_dim=1):
        super(Critic, self).__init__()

        # 定义权重和偏置
        self.W1 = nn.Parameter(torch.FloatTensor(embedding_length, 128).uniform_(-0.5, 0.5))  # 768,128
        self.W2 = nn.Parameter(torch.FloatTensor(128, 128).uniform_(-0.5, 0.5))  # 128,128
        self.W3 = nn.Parameter(torch.FloatTensor(num_teacher, 128).uniform_(-0.5, 0.5))  # 2,128
        self.b = nn.Parameter(torch.FloatTensor(1, 128).uniform_(-0.5, 0.5))

        layers = []
        in_dim = input_dim
        for h_dim in hidden_dim:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, output_dim))
        self.model = nn.Sequential(*layers)
        self.optimizer = optim.Adam(self.parameters(), lr=0.0003)

    def forward(self, x1, x2, x3):
        # x1: (num_sent, batch_size, embedding_length) 2,128,768
        # x2: (num_teacher, batch_size, batch_size) 2,128,128
        # x3: (num_teacher, 1) 1,2

        # 将输入与相应的权重相乘
        x1_ = torch.matmul(x1, self.W1)  # 2,128,128
        x2_ = torch.matmul(x2, self.W2)  # 2,128,128

        # 将第三个输入乘以其权重
        x3_ = torch.matmul(x3, self.W3)  # 1,128

        # 将所有结果相加并加上偏置
        scaled_out = torch.sigmoid(x1_ + x2_ + x3_ + self.b)
        # 限制输出范围并保证它是1x1维度
        scaled_out = torch.clamp(scaled_out, min=1e-5, max=1 - 1e-5)  # 2,128,128

        # 重塑 scaled_out 以匹配全连接层的期望输入维度
        scaled_out_reshaped = scaled_out.view(-1, 128)  # 形状现在是 [-1, 128]

        # 将重塑后的结果传递给 Critic 网络的模型部分
        critic_out = self.model(scaled_out_reshaped)
        critic_out = critic_out.mean()
        return critic_out
