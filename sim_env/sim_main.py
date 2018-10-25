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
            eps_decay = 1, 
            eps_min = 0.01, 
            l_rate = 0.001
            )

#Train agent
episodes = 150000
steps = 200
goalCounter = 0
stepCounter = 0
avg = -1
ep = 0

for ep in range(episodes):
    ep += 1
    state = env.reset()

    for step in range(steps):
        action = int(agent.action(state))
        
        next_state, reward, done = env.step(action) 

        #agent.remember(state, action, reward, next_state, done)

        state = next_state

        if done:
            if reward > 1: 
                goalCounter += 1
                stepCounter += step
                avg = stepCounter / goalCounter

            break

    if ep % 1000 == 0:    
        print("episode {} done, goal found {} times with an avg steps of {}".format(ep, goalCounter, avg))
    
    #agent.replay(64)
    #agent.save_model()