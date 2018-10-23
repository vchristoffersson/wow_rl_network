from env import Env
from agent import Agent

#get state_size and action_size from env?
env = Env()

state_size = env.get_state_size()
action_size = env.get_action_size()

agent = Agent(
            state_size = state_size, 
            action_size = action_size, 
            discount = 0.8, 
            eps = 0.98, 
            eps_decay = 0.9, 
            eps_min = 0.01, 
            l_rate = 0.001
            )

#Train agent
episodes = 1000
steps = 500

for ep in range(episodes):
    state = env.reset()

    for step in range(steps):
        action = int(agent.action(state))

        next_state, reward, done = env.step(action)
        
        #print("reward: {}, done: {}, step: {}".format(reward, done, step))

        agent.remember(state, action, reward, next_state, done)

        state = next_state

        if done:
            print("episode {}/{} done, after {} steps with reward {}".format(ep, episodes, step, reward))
            break

    agent.replay(100)