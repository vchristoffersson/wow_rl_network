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
            eps_decay = 0.995, 
            eps_min = 0.01, 
            l_rate = 0.0001
            )

#Train agent
episodes = 500
steps = 1000
ep = 0
goalCounter = 0
for ep in range(episodes):
    ep += 1
    state = env.reset()

    for step in range(steps):
        action = int(agent.action(state, ep))

        next_state, reward, done = env.step(action)   
        
        agent.remember(state, action, reward, next_state, done)

        state = next_state

        if done:
            if reward > 1: 
                goalCounter += 1
                print("GOAL REACHED")

            print("episode {}/{} done, after {} steps".format(ep, episodes, step))
            break

    agent.replay(200)

print('Goal was reached {} times'.format(goalCounter))