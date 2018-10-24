from sim_env import Env
from sim_agent import Agent

#get state_size and action_size from env?
env = Env()

state_size = 5
action_size = 4

agent = Agent(
            state_size = state_size, 
            action_size = action_size, 
            discount = 0.9, 
            eps = 1, 
            eps_decay = 0.9999, 
            eps_min = 0.01, 
            l_rate = 0.00001
            )

#Train agent
episodes = 50000
steps = 100
goalCounter = 0

for ep in range(episodes):
    state = env.reset()

    for step in range(steps):
        action = int(agent.action(state))

        #print("action: {}".format(action))

        next_state, reward, done = env.step(action)
        
        #print("reward: {}, done: {}, step: {}, action: {} ".format(reward, done, step, action))

        agent.remember(state, action, reward, next_state, done)

        state = next_state

        if done:
            if reward > 1: 
                goalCounter += 1
                print("episode {} done, after {} steps, reward: {}, goal #{}".format(ep, step, reward, goalCounter))
            break

    agent.replay(64)
    
agent.save_model()