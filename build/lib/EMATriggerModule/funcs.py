import csv
import numpy as np
import pandas as  pd
import requests

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Input
from tensorflow.keras.optimizers import Adam
import tensorflow as tf


from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory
from sklearn.ensemble import RandomForestClassifier as RF



def load_labels():    
    url = "https://unite.healthscitech.org/api/prompt/621987869b9e9ba69b4e0776/submission?page=0&per_page=10000"
    
    payload={}
    headers = {
      'Authorization': 'WyI1ZWUxOTJlZTc5YjRkZGQ5NjIzMWM1M2MiLCIkNSRyb3VuZHM9NTM1MDAwJEdtd1hqRTlyWFYzNkdpQ3IkVHRkM1Qvd2tnL1lNT0FXQVZtU2FCa2FZMGVYZ0NlU2lVUnhwQ24yZEJ2MyJd.Yiz4fA.6K19Vrh3O0hN0fAl9ner0oLnROQ',
      'Cookie': 'session=.eJwljktqBDEMBe_i9RAs62OpL9PIskyyG7pnViF3j2E2Dx4UVP2Wc115f5fjdb3zUc6fWY5SeTUlcm-py3mMqFW6dUv2ihzhrj0xKYF9ccxpvCZEUMzMkQNzIoG6OIKKVSER6HsHsiUasZPOYdOgLpclHbNmb0OHdK1lh7zvvD41nAnWMrsNmlslDSEYY1PrPiM2cufr857u5QChDkAo-CUqgCp__6StQWo.Yiz4fA.zPV0A1YKBVzFx_O02Ibq6D4sgF8'
    }
    
    users_dict = {'6219a83c5e0c6270fe67f573': 'unite3rct444',
                  '621c25fdce0e2e1f2425d452': 'unite3rct454',
                  '621c261be8fd6d2087f1d654': 'unite3rct464',
                  '621c2630fdda58cf82f06788': 'unite3rct474',
                  '621c264ace0e2e1f2425d454': 'unite3rct484',
                  '621c2661e8fd6d2087f1d656': 'unite3rct494',
                  '621c26744090fad1d9e09f0a': 'unite3rct504',
                  '621c2692fdda58cf82f06789': 'unite3rct514',
                  '621c26a7e8fd6d2087f1d657': 'unite3rct524',
                  '621c26c0fdda58cf82f0678a': 'unite3rct534',
                  '621c26d64090fad1d9e09f0b': 'unite3rct544'}
    
    response = requests.request("GET", url, headers=headers, data=payload)
    response = pd.json_normalize(response.json())
    response['data.stress_range_microrctp3_last_modify'].fillna(response['data.start_time'], inplace=True)
    response = response[['user.$oid', 'data.stress_range_microrctp3', 'data.stress_range_microrctp3_last_modify']]
    response.rename(columns={'data.stress_range_microrctp3_last_modify': 'timestamp_label', 'user.$oid': 'user', 
                             'data.stress_range_microrctp3':'data.stressed'}, inplace=True)
    response['user'].replace(users_dict, inplace=True)
    return response


def reject_outliers(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s<m]
    

def build_samples_file(path, columns):
    with open(path, 'a', newline='') as file:
        file_writer = csv.writer(file, delimiter=',')
        file_writer.writerow(columns)
    

def build_stress_model(n=13, out_nodes=1, norm_layer=None, hidden_layers = [25,40,25,10,3], activation='relu'):
    model = Sequential()
    if norm_layer:
        model.add(norm_layer)
        
    model.add(Input(shape=(n,)))
    for h in hidden_layers:
        model.add(Dense(h,activation=activation))
    model.add(Dense(out_nodes, activation= 'sigmoid'))
    return model

def RF_stress_model(subj_data, offline_data, features, 
                    labels_conv_dict = {'not at all': 0, 'a little bit': 0, 'some':0, 'a lot':1, 'extremely':1}):
    subj_data = subj_data[subj_data['reported_stress'].isin(labels_conv_dict.keys())].replace(labels_conv_dict)
    clf = RF(n_estimators = 500, max_depth = 5, class_weight= 'balanced')
    X_tr = offline_data[features].append(subj_data[features])
    y_tr = offline_data['reported_stress'].append(subj_data['reported_stress'])
    clf.fit(X_tr, y_tr)
    return clf

def build_action_model(n=3, a=2, h= [5,9,7,5], activation='relu'):
    #n: dimention of space
    #a: number of actions
    #h: number of node s in each hidden layer 
    model = Sequential()
    #model.add(Input(shape=(n,)))
    model.add(Flatten(input_shape=(1, n))) 
    #model.add(Input(shape = (n,)))
    for i in h:
        model.add(Dense(i, activation=activation)) 
    model.add(Dense(a, activation='linear'))
    return model

def build_agent(model, actions=2):
    #policy = BoltzmannQPolicy()
    #policy = EpsGreedyQPolicy(eps=0.2)
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=.3, value_min=.1, value_test=.05, nb_steps=1000)
    memory = SequentialMemory(limit=50000, window_length = 1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, nb_actions=actions, gamma=0.95, nb_steps_warmup=100, target_model_update=1e-2) 
    return dqn

def train_agent(env, hs = [5,9,7,5], steps=3000):
    model_agent = build_action_model(h = hs)
    agent = build_agent(model_agent)
    agent.compile(Adam(lr=1e-3), metrics=['mae'])
    train_history = agent.fit(env, nb_steps = steps, visualize = False, verbose=1)
    
    return agent, train_history

def build_agent_forload(env, hs = [5,9,7,5]):
    model_agent = build_action_model(h = hs)
    agent = build_agent(model_agent)
    agent.compile(Adam(lr=1e-3), metrics=['mae'])
    return agent

def action_pred(agent, sample):
    state = sample.copy()
    while len(state)==1:
        state = state[0]
    observation = agent.memory.get_recent_state(state) 
    q_values = agent.compute_q_values(observation)
    action = agent.test_policy.select_action(q_values=q_values)
    
    return action, q_values

def agent_divergence_test(agent, states):
    actions = []
    for state in states:
        actions.append(action_pred(agent, state)[0])
    num_trigs = sum(actions)
    flag = num_trigs < 0.05*len(actions) or  num_trigs > 0.3*len(actions)
    return flag

def last_trigger_time(data):
    if data.query('triggered==1').empty:
        return -100
    else:
        idx = data.query('triggered==1').index.max()
        return data.timestamp2_sample.loc[idx]

def res_rate_update(res_r, labeled_data, alpha=0.2):
    labeled_data = labeled_data[labeled_data['resp_t_min']>0]
    labeled_data = labeled_data[['sample_time_hour','resp_t_min']]
    labeled_data['responsive'] = (labeled_data['resp_t_min']<32).astype('int')
    res_r_subj = np.ones(24)*(-1)
    for i in range(24):
        if any(labeled_data['sample_time_hour']==i):
            res_r_subj[i] = labeled_data[labeled_data['sample_time_hour']==i]['responsive'].mean()
    updated_res = (1-alpha)*res_r + alpha*((res_r_subj==-1)*res_r + (res_r_subj!=-1)*res_r_subj)
    
    return updated_res

def ftune_s_model(model_stress, labeled_data, features, 
                  labels_conv_dict = {'not at all': 0, 'a little bit': 0, 'some':0, 'a lot':1, 'extremely':1}):
    labeled_data = labeled_data[labeled_data['reported_stress'].isin(labels_conv_dict.keys())]
    weights = np.ones(labeled_data.shape[0])
    weights[:int(len(weights)*0.8)] = np.linspace(.1,1,int(len(weights)*0.8))
    model_stress.fit(labeled_data[features], labeled_data['reported_stress'].replace(labels_conv_dict), epochs=10, sample_weight=weights, verbose=0)
    return model_stress