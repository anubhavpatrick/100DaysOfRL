#!/usr/bin/env python
# coding: utf-8

# In[1]:


#May require to install opencv if it is not available: conda install -c menpo opencv

#REMEMBER: Open CV used BGR image encoding format
import cv2

import numpy as np 
from PIL import Image
import pickle #to save and load python objects
import matplotlib.pyplot as plt
from matplotlib import style 
import time
import os


# In[3]:


style.use('ggplot')


# In[6]:


#Setting the constants
SIZE = 10 #Grid size of environment will be 10*10
EPISODES = 25000
MOVE_PENALTY = 1
ENEMY_PENALTY = 300
FOOD_REWARD = 25
epsilon = 1 #epsilon greedy policy
EPS_DECAY = 0.9998
SHOW_EVERY = 3000
start_q_table = None #can be prior q table saved using pickle library
LEARNING_RATE = 0.1
DISCOUNT = 0.95


# In[7]:


PLAYER_N=1
FOOD_N=2
ENEMY_N=3
#open cv using BGR format for image representation
d={1:(255,0,0),
   2:(0,255,0),
   3:(0,0,255)
}


# In[ ]:


#Creating class for player, food and enemy
class Blob:
    def __init__(self): #Python constructor
        self.x = np.random.randint(0,SIZE)
        self.y = np.random.randint(0,SIZE)
        #To Do: There can be a issue in which enemy and/or player and/or food land on the same cell
    
    #for debugging purposes
    def __str__(self):
        return f"{self.x}, {self.y}"
    
    #operator overloading
    def __sub__(self,second):
        return (self.x - second.x, self.y-second.y)
        
    #defining possible actions by Blob agent
    def action(self,choice): #very simplified discrete action space consisting of four actions only
        if choice == 0:
            self.move(x=1,y=1)
        elif choice==1:
            self.move(x=-1,y=-1)
        elif choice ==2:
            self.move(x=-1,y=1)
        elif choice ==3:
            self.move(x=1,y=-1)
        elif choice == 4:#Move right
            self.move(x=1,y=0)
        elif choice ==5 :#Move left
            self.move(x=-1,y=0)
        elif choice ==6: #Move Up
            self.move(x=0,y=1)
        elif choice ==7: #Move Down
            self.move(x=0,y=-1)
        #To Do: Add more choices
    
    def move(self, x=False,y=False):
        #The agent will move either randomly or based on value passed in x or y
        #if not x:#x is local var and self.x is class var
        #self.x = np.random.randint(-1,2)  
        #else:
        self.x += x
            
        #if not y:
        #    self.y = np.random.randint(-1,2)  
        #else:
        self.y += y
        
        #We have to also ensure that blob does not move outside the boundaries
        if self.x < 0:
            self.x = 0
        elif self.x > (SIZE-1):
            self.x = SIZE-1
        
        if self.y < 0:
            self.y = 0
        elif self.y > (SIZE-1):
            self.y = SIZE-1


# In[ ]:


#States of our Q Table consist of difference between x and y coordinates of the player and food Blob AND player and enemy Blob
if start_q_table is None:
    q_table = {} #dictionary
    for x1 in range(-SIZE+1, SIZE):
        for y1 in range(-SIZE+1, SIZE):
            for x2 in range(-SIZE+1, SIZE):
                for y2 in range(-SIZE+1,SIZE):
                    q_table[((x1,y1),(x2,y2))]= [np.random.uniform(-5,0) for i in range(8)] #since there are eight discrete actions
                    #The initial values need to be modified to see the impact
                    
else: #The Q Table exists and may be present in the form of pickle object
    with open(start_q_table,"rb") as f:
        q_table = pickle.load(f)


# In[ ]:


#a simple function to display the state of environment
from IPython.display import clear_output
import time
def display(player,food,enemy):
    os.system('clear')
    env_list = [[None, None, None, None, None, None, None, None, None, None],
                [None, None, None, None, None, None, None, None, None, None],
                [None, None, None, None, None, None, None, None, None, None],
                [None, None, None, None, None, None, None, None, None, None],
                [None, None, None, None, None, None, None, None, None, None],
                [None, None, None, None, None, None, None, None, None, None],
                [None, None, None, None, None, None, None, None, None, None],
                [None, None, None, None, None, None, None, None, None, None],
                [None, None, None, None, None, None, None, None, None, None],
                [None, None, None, None, None, None, None, None, None, None]]

    for i in range(0,SIZE):
        for j in range(0,SIZE):
            '''print(f'i:{i} j:{j}')
            print(f'Player {i==player.x and j==player.y}')
            print(f'Food {i==food.x and j==food.y}')
            print(f'Enemy {i==enemy.x and j==enemy.y}')'''
            env_list[i][j]= ' '
            if ((i==player.x) and (j==player.y)):
                env_list[i][j]='P'
            if ((i==food.x) and (j==food.y)):
                env_list[i][j]='F'
            if ((i==enemy.x) and (j==enemy.y)):
                env_list[i][j]='E'
    return env_list
'''p=Blob()
f=Blob()
e=Blob()
print(f'Player {p.x} {p.y}')
print(f'Food {f.x} {f.y}')
print(f'Enemy {e.x} {e.y}')'''
'''temp=display(p,f,e)
for i in range(SIZE):
    print(temp[i])'''


# ### Q Learning Algorithm

# In[ ]:


START_EPSILON_DECAYING = 1#From which episode we want to start to decay epsilon
END_EPSILON_DECAYING = EPISODES // 2 #Till which episode we want to decay epsilon
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)
episode_rewards = []
for episode in range(EPISODES):
    player = Blob()
    food = Blob()
    enemy = Blob()
    
    if episode % SHOW_EVERY == 0:
        print(f'on #{episode}, epsilon: {epsilon}')
        print(f'{SHOW_EVERY} ep mean {np.mean(episode_rewards[-SHOW_EVERY: ])}')
        show = True
    else:
        show = False
    
    #updating epsilon value
    
    episode_reward = 0
    for i in range(200): #Here 200 is the steps in each episode. It is a hyperparameter
        obs = (player - food, player - enemy)#Remember the function overriding
        if np.random.random() > epsilon:
            action = np.argmax(q_table[obs])
        else:
            action = np.random.randint(0,8)
        
        player.action(action)
       
        if episode>24000:
            env_state = display(player, food, enemy)
            #time.sleep(0.5)#delay in sec
            '''print(f'Player {player.x} {player.y}')
            print(f'Food {food.x} {food.y}')
            print(f'Enemy {enemy.x} {enemy.y}')'''
            print(f'Episode: {episode} Step: {i}',end='')
            print(f'Mean Reward{np.mean(episode_rewards[-SHOW_EVERY: ])}')
            for i in range(SIZE):
                print('|',end='')
                for j in range(SIZE):
                    print(f' {env_state[i][j]}|',end='')
                print('') 
        #Deciding the reward or penalty of agent after every step in each episode
        if player.x == enemy.x and player.y == enemy.y:
            reward = -ENEMY_PENALTY
        elif player.x == food.x and player.y == food.y:
            reward = FOOD_REWARD
        else:
            reward = -MOVE_PENALTY
            
        new_obs = (player-food,player-enemy)
        max_future_q = np.max(q_table[new_obs])
        current_q = q_table[obs][action]
    
        if reward == FOOD_REWARD:
            new_q = FOOD_REWARD
            q_table[obs][action] = new_q
            episode_rewards.append(reward)
            break
        elif reward == -ENEMY_PENALTY:
            new_q = -ENEMY_PENALTY
            q_table[obs][action] = new_q
            episode_rewards.append(reward)
            break
        else:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[obs][action] = new_q
            episode_rewards.append(reward)
    
    # Decaying is being done every episode if episode number is within decaying range
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

