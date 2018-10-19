from env import Env
from agent import Agent

#get state_size and action_size from env?
env = Env()

state_size = None
action_size = 4

agent = Agent(
            state_size = state_size, 
            action_size = action_size, 
            discount = 0.9, eps = 1.0, 
            eps_decay = 0.995, 
            eps_min = 0.01, 
            l_rate = 0.001
            )


#Train agent
episodes = 500

for ep in range(episodes):
    state = env.reset()

    for steps in range(500):
        action = agent.action(state)

        #health percentage not needed?
        ret = env.step(action).contents
        next_state = ret[0]
        health = ret[1]
        reward = ret[2]
        terminal = ret[3]

        agent.remember(state, action, reward, next_state, terminal)

        state = next_state

        if terminal:
            print("episode {}/{} done, after {} steps".format(ep, episodes, steps))
            break

    agent.replay(32)