import numpy as np
from env import Env
from keras.utils.np_utils import to_categorical as one_hot
from collections import namedtuple
from dqn_model import DoubleQLearningModel, ExperienceReplay
import random

def eps_greedy_policy(q_values, eps):
    '''
    Creates an epsilon-greedy policy
    :param q_values: set of Q-values of shape (num actions,)
    :param eps: probability of taking a uniform random action 
    :return: policy of shape (num actions,)
    '''
    # Complete this function
    policy = np.zeros(q_values.shape)
    rand = random.uniform(0, 1)
    
    if rand < eps:
        policy[:] = 1 / len(policy)
    
    else:
        best_action = np.argmax(q_values)
        policy[best_action] = 1
        
    return policy

def calculate_td_targets(q1_batch, q2_batch, r_batch, t_batch, gamma=.9):
    '''
    Calculates the TD-target used for the loss
    : param q1_batch: Batch of Q(s', a) from online network, shape (N, num actions)
    : param q2_batch: Batch of Q(s', a) from target network, shape (N, num actions)
    : param r_batch: Batch of rewards, shape (N, 1)
    : param t_batch: Batch of booleans indicating if state, s' is terminal, shape (N, 1)
    : return: TD-target, shape (N, 1)
    '''
    
    # Complete this function
    Y = np.zeros(r_batch.shape)
    N = len(r_batch)
    
    Y = np.zeros(r_batch.shape)
    
    for i in range(N):
        if t_batch[i]:
            Y[i] = r_batch[i]
        else:
            Y[i] = r_batch[i] + gamma * q2_batch[i, np.argmax(q1_batch[i])]

    return Y

def train_loop_ddqn(model, env, num_episodes, batch_size=64, gamma=.94):        
    Transition = namedtuple("Transition", ["s", "a", "r", "next_s", "t"])
    eps = 1.
    eps_end = .1 
    eps_decay = .001
    R_buffer = []
    R_avg = []
    for i in range(num_episodes):
        state = env.reset() #reset to initial state
        #state = np.expand_dims(state, axis=0)/2
        terminal = False # reset terminal flag
        ep_reward = 0
        q_buffer = []
        steps = 0
        while not terminal:
            #env.render() # comment this line out if you don't want to / cannot render the environment on your system
            steps += 1
            q_values = model.get_q_values(state)
            q_buffer.append(q_values)
            policy = eps_greedy_policy(q_values.squeeze(), eps) 
            action = int(np.random.choice(num_actions, p=policy)) # sample action from epsilon-greedy policy
            new_state, reward, terminal = env.step(action) # take one step in the evironment
            #new_state = np.expand_dims(new_state, axis=0)/2
            
            # only use the terminal flag for ending the episode and not for training
            # if the flag is set due to that the maximum amount of steps is reached 
            t_to_buffer = terminal if not steps == 200 else False
            
            # store data to replay buffer
            replay_buffer.add(Transition(s=state, a=action, r=reward, next_s=new_state, t=t_to_buffer))
            state = new_state
            ep_reward += reward
            
            # if buffer contains more than 1000 samples, perform one training step
            if replay_buffer.buffer_length > 1000:
                s, a, r, s_, t = replay_buffer.sample_minibatch(batch_size) # sample a minibatch of transitions
                q_1, q_2 = model.get_q_values_for_both_models(np.squeeze(s_))
                td_target = calculate_td_targets(q_1, q_2, r, t, gamma)
                model.update(s, td_target, a)    
                
        eps = max(eps - eps_decay, eps_end) # decrease epsilon        
        R_buffer.append(ep_reward)
        
        # running average of episodic rewards
        R_avg.append(.05 * R_buffer[i] + .95 * R_avg[i-1]) if i > 0 else R_avg.append(R_buffer[i])
        print('Episode: ', i, 'Reward:', ep_reward, 'Epsilon', eps, 'mean q', np.mean(np.array(q_buffer)))
        
        model.save_weights()

        # if running average > 195, the task is considerd solved
        if R_avg[-1] > 195:
            return R_buffer, R_avg
    return R_buffer, R_avg

env = Env()

num_actions = env.get_action_size()
obs_dim = env.get_state_size()

# Our Neural Netork model used to estimate the Q-values
model = DoubleQLearningModel(state_dim=obs_dim, action_dim=num_actions, learning_rate=1e-4)

# Create replay buffer, where experience in form of tuples <s,a,r,s',t>, gathered from the environment is stored 
# for training
replay_buffer = ExperienceReplay(state_size=obs_dim)

# Train
num_episodes = 1200 
batch_size = 128 
R, R_avg = train_loop_ddqn(model, env, num_episodes, batch_size)