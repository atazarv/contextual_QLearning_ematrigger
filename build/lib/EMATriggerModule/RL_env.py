import gym
import random
import numpy as np
from gym import Env
from gym import spaces
from gym.spaces import Discrete, Box
from gym.utils import seeding
from datetime import datetime


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
#from tensorflow.keras import Input
from tensorflow.keras.optimizers import Adam
import tensorflow as tf


#%%
def reward_F(td):
    #b:  number of sample intervals that the reward is 1 after that
    return (td<1)*(td-1)+1-(td<0)*(td)

def last_trigger_time(data):
    if data.query('triggered==1').empty:
        return -100
    else:
        idx = data.query('triggered==1').index.max()
        return data.timestamp2_sample.loc[idx] 

def update_state(data, new_sample, response_rate):
    PP = new_sample.NN_pred_stress+new_sample.RF_pred_stress
    RR = response_rate[new_sample.sample_time_hour]
    last_trig_minutes = (new_sample.timestamp2_sample - last_trigger_time(data))/6e4
    F = np.clip(last_trig_minutes/15/12,0,1)
    n_state = np.asarray([PP, RR, F]).astype('float32')#.reshape(1,-1)
    #n_state = tf.convert_to_tensor([D, T, F], dtype=tf.float32)
    return n_state  
    

#%%
class AL(Env):
    def __init__(self, data, response_rate):
        self.action_space = Discrete(2)
        self.observation_space = Box(low = np.array([0,0,0], dtype = 'float32'), high = np.array([1,1,1], dtype = 'float32'))
        self.data = data
        self.cur_tstep = 0
        self.state = None #(0,0,0)
        self.response_rate = np.array(response_rate, dtype='float32')
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    def step(self, action):
        self.cur_tstep+=1
        n_state = update_state(self.data.iloc[:self.cur_tstep], self.data.iloc[self.cur_tstep], self.response_rate)
        reward = self._reward(action, n_state, b=1) 
        self.state = n_state
        done = self.cur_tstep >= len(self.data)-1
        if done:
            self.cur_tstep=0
        info = {}
        return n_state, reward, done, info
    
    def render(self):
        pass
    
    def reset(self):
        self.cur_tstep = 0
        first_samp = self.data.iloc[0]
        D = first_samp.NN_pred_stress + first_samp.RF_pred_stress
        RR = self.response_rate[first_samp.sample_time_hour]
        F = 1
        self.state = np.asarray([D,RR,F]).astype('float32')
        return self.state
    
    def _reward(self, action, n_state, b=10):
        #r0 = 1/(b*n_state[0]+1)     # 0 < r3 <=1
        #r0 = int(n_state[0]>.5)         #pred_prob
        r0 = 1/(1+np.exp(-20*(n_state[0]-.5)))
        #r1 =  n_state[1]            # response_rate 0 <= r2 <= 1
        r1 = reward_F(n_state[1])   # distance_to_last_trigger 0 < r1 <=  1
        r2 = 1/(1+np.exp(-10*(n_state[2]-0.5)))
        reward_p = r0 + 2*r1 + r2
        #simp_reward = int(n_state[2]>0.5 and n_state[1]>0.4)
        if action:
            #return (n_state[1]>0.25 and r2>.2)*((reward_p>2)*5 + (reward_p<=2)*reward_p)
            #return (r2>.2)*((reward_p>2)*5 + (reward_p<=2)*reward_p/2)+(r2<=0.2)*(-1)
            #return reward_p
            #return 10*simp_reward-1
            return reward_p
        else:
            #if n_state[1]<0.25 or r2<.2:
            #return (r2<0.2)*5 + (r2>=0.2)*(reward_p<2)*(2-reward_p)
            #return 4-reward_p
            #return -2*simp_reward+1
            return 3 - reward_p
        
#%%
#env = AL(labeled_data, response_rate)
#
#state = env.reset()
#done = False
#action = np.random.randint(0,2,200)
#for i in range(98):
#    next_state, reward, done, info = env.step(action[i])
#    print(f" state: {state}, next state:{next_state}, reward: {reward}")
#    if done:
#        print(f"end session {next_state},{state}")
#    state = next_state