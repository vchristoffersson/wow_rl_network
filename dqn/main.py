from env import Env
from agent import Agent

#get state_size and action_size from env?
env = Env()

state_size = env.get_state_size()
action_size = env.get_action_size()

agent = Agent(
            state_size = state_size, 
            action_size = action_size, 
            discount = 0.9, 
            eps = 1.0, 
            eps_decay = 0.99, 
            eps_min = 0.01, 
            l_rate = 0.0001
            )

#Train agent
episodes = 1000
steps = 1000

for ep in range(episodes):
    state = env.reset()

    for step in range(steps):
        action = int(agent.action(state))

        next_state, reward, done = env.step(action)
        
        #print("reward: {}".format(reward))

        agent.remember(state, action, reward, next_state, done)

        state = next_state

        if done:
            print("episode {}/{} done, after {} steps".format(ep, episodes, step))
            break

    agent.replay(200)