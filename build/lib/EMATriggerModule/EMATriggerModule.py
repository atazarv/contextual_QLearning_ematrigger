# import sys
# sys.path.append('.')
# sys.path.append('D:/UCI/7th year/Active Learning Engine')
# #%%
import csv
import numpy as np
import os
from pathlib import Path
import pandas as  pd
from datetime import datetime
import shutil


from tensorflow import keras
from tensorflow.keras import backend as K

import joblib

#**************************************************************
#import os
#os.chdir('D:/UCI/7th year/Active Learning Engine/EMATrigger/EMATriggerModule')
#from Feature_Extraction import PPGFeatureExtraction
#from Sanity_Check import SanityCheck
#from Label_Matching import LabelMatching
#from RL_env import AL
#from funcs import *

from .Feature_Extraction import PPGFeatureExtraction
from .Sanity_Check import SanityCheck
from .Label_Matching import LabelMatching
from .RL_env import AL
from .funcs import *

#%%
class TriggerModule():
    def __init__(self):
        pass

    def main(self, datapath, filespath, user_id, realtime=True, sleep = False): 
        raw_data = pd.read_csv(datapath, header=0, delimiter='\t')
        check_mod = SanityCheck()
        check_mod.check_signal(raw_data, signal_duration = [2,5])

        feat_extr = PPGFeatureExtraction(raw_data)
        features, ppg_features = feat_extr.FeatureExtract()


        #rest_time = datetime.strptime(str(datapath)[-23:-4], '%Y-%m-%d-%H-%M-%S').hour < 7
        #features =  ['bpm', 'ibi', 'sdnn', 'sdsd', 'rmssd', 'pnn20', 'pnn50', 'hr_mad', 'sd1', 'sd2', 's', 'sd1/sd2', 'breathingrate']
        header =   ['timestamp1_sample', 'timestamp2_sample', 'user'] + features + ['triggered', 'random_action', 'sample_time_hour', 
                    'NN_pred_stress', 'RF_pred_stress', 'filename', 'realtime', 'sleep', 'timestamp_label', 'resp_t_min', 'reported_stress']
        
        ##  INITIALIZIATION
        if not((filespath / user_id).exists()):
            (filespath/user_id).mkdir()
            build_samples_file(filespath / user_id / ('Sample_'+user_id+'.csv'), header)
            shutil.copytree(filespath / "BINARY_STRESS_DETECTOR_DEFAULT" , filespath / user_id / ("stress_detector_"+user_id))
            shutil.copytree(filespath / "AGENT_DEFAULT_trnedondatafromallusers" , filespath / user_id / ("agent_"+user_id))
            shutil.copy(filespath / 'RESP_RATE_DEFAULT.csv', filespath / user_id / ("resp_rate_"+user_id+".csv"))
            shutil.copy(filespath / 'RF_CLASSIFIER_DEFAULT.joblib', filespath / user_id / ('rf_stress_detector_'+user_id+'.joblib'))


        ##LOAD STRESS DETECTOR AND DETECT STRESS
        NN_model_stress = keras.models.load_model(filespath / user_id / ("stress_detector_"+user_id))
        RF_model_stress = joblib.load(filespath / user_id / ('rf_stress_detector_'+user_id+'.joblib'))

        samp = np.array(ppg_features).reshape(1,-1)
        NN_pred_st = NN_model_stress.predict(samp)[0,0]
        RF_pred_st = RF_model_stress.predict_proba(samp)[0,1]
        #confidence = 2*abs(pred_st-0.5)

        #LOAD PROCESSED DATA, PARTIALLY LABELED
        SS = pd.read_csv(filespath / user_id / ('Sample_'+user_id+'.csv'))
        SS.rename(columns = {'resp_time': 'resp_t_min'}, inplace=True)

        sc = SS.shape[0]
        sample_h = datetime.fromtimestamp(raw_data.timestamp.iloc[-1]/1000).hour
        response_rates = np.loadtxt(filespath / user_id / ("resp_rate_"+user_id+".csv"), ndmin=2, delimiter=',')
        action = 0
        random_action= 0
        
        if not(sc%100) and sc:
            #update the labeled data
            #labels = pd.read_csv(filespath / "labels.csv")
            labels = load_labels() 
            match_mod = LabelMatching()
            labeled_data = match_mod.match(SS, labels, header, stress_labels = [1,2,3,4,5])
            labeled_data.to_csv(filespath / user_id / ("Sample_"+user_id+".csv"), index=False)
            #update stress detector
            NN_model_stress = ftune_s_model(NN_model_stress, labeled_data, features, labels_conv_dict={1:0, 2:0, 3:0, 4:1, 5:1})
            NN_model_stress.save(filespath / user_id / ('stress_detector_'+user_id))
            
            offline_data = pd.read_csv(filespath / 'labeled_data.csv')
            RF_model_stress = RF_stress_model(labeled_data, offline_data, features, labels_conv_dict={1:0, 2:0, 3:0, 4:1, 5:1})
            
            RF_pred_st = RF_model_stress.predict_proba(samp)[0,1]
            NN_pred_st = NN_model_stress.predict(samp)[0,0]
            
            #conf_NN = 2*abs(pred_st-0.5)
            #update response time
            response_rate = res_rate_update(response_rates[:,-1], labeled_data, alpha = .2)
            response_rates = np.hstack((response_rates, response_rate.reshape(-1,1)))
            np.savetxt(filespath / user_id / ("resp_rate_"+user_id+".csv"), response_rates, delimiter=',')
            #build and train agent
            labeled_data['NN_pred_stress'] = NN_model_stress.predict(labeled_data[features]).T[0]#.numpy().T[0]
            labeled_data['RF_pred_stress'] = RF_model_stress.predict_proba(labeled_data[features])[:,1]
            #labeled_data['confidence'] = 2*abs(labeled_data['pred_stress']-0.5)
            #labeled_data['confidence'] = [0]*len(labeled_data)
            #ind = np.clip(sc, 0, 500)
            #environment = AL(labeled_data.iloc[-ind:], response_rate)
            environment = AL(labeled_data, response_rate)
            environment.reset()
            agent = build_agent_forload(environment)
            agent.load_weights(filespath / user_id / ("agent_"+user_id) / "agent")
            agent.fit(environment, nb_steps = 5*len(environment.data), visualize = False, verbose=1)
            agent_test_data = pd.read_csv(filespath / 'states.csv')
            agent_div_flag = agent_divergence_test(agent, agent_test_data.values)
            if not agent_div_flag:
                agent.save_weights(filespath / user_id / ("agent_"+user_id) / "agent", overwrite=True)
                print('*** Stress Model Updated -- Response Rate Updated -- Agent Updated ***')
            else:
                print('*** Stress Model Updated -- Response Rate Update ***')
            
            
        
        if np.random.random()<0.05:
            action = 1
            random_action = 1
        #take action based on the most recent agent
        else:
            response_rate = response_rates[:,-1]
            K.clear_session()
            agent = build_agent_forload(0)
            agent.load_weights(filespath / user_id / ("agent_"+user_id) / "agent")
            last_trig_dist = (raw_data.timestamp.iloc[-1] - last_trigger_time(SS))/6e4
            last_trig_dist = np.clip(last_trig_dist/15/12, 0,1)
            state = np.array([NN_pred_st+RF_pred_st, response_rate[sample_h], last_trig_dist], dtype='float32').reshape(1,1,-1)
            action = action_pred(agent, state)[0]
                
        
        locs = str(datapath).find('data_'+user_id)
        
        Sample = [raw_data.iloc[0].timestamp, raw_data.iloc[-1].timestamp, user_id] + ppg_features + [action, random_action,
                 sample_h, NN_pred_st, RF_pred_st, str(datapath)[locs:], int(realtime), int(sleep)] + [0,0,0]

        with open(filespath / user_id / ('Sample_'+user_id+'.csv'), 'a', newline='') as file:
            file_writer = csv.writer(file, delimiter=',')
            file_writer.writerow(Sample)
        
        return action



#%%
if __name__ == "__main__":
    
    #datapath = Path(r'D:\UCI\Unite\Unite_RCT\Source Data\Raw Data 2 minutes')
    #datapath = Path(r'D:\UCI\Unite\Unite_RCT\Source Data\data_uniterct148-2021-01-29-14-34-46')
    datapath = Path(r'D:\New Foloder\Raw_data')
    filespath = datapath.parent #.parents[i]
    filespath.mkdir(exist_ok = True)
    files = os.listdir(datapath)
    user_id = 'unite3rct444'
    
    try:
#        os.remove(filespath / ("Sample_"+user_id+".csv")) 
#        os.remove(filespath / (user_id+"_resp_rate.csv"))
        shutil.rmtree(filespath / (user_id))
    except:
        pass
        
    files = [files[i] for i in range(len(files)) if (files[i][-4:]=='.csv') and (files[i][5:-24]==user_id)]
    q=0; p=0; r = 0; s=0
    start = datetime.now()

    ema_mod = TriggerModule()
    for f in files[:150]:
        K.clear_session()
        if q%100==0:
            print(q)

        user_id = f[f.find('unite3rct') : f.find('unite3rct')+12]
        q+=1
        try:
            trig = ema_mod.main(datapath=datapath / f, filespath= filespath, user_id=user_id)
            if trig: 
                r+=1
                #print(f, trig) 
                 
        except ValueError:
            p+=1
            pass 
        except:
            s+=1
            #traceback.print_exec()

    print("\n \n", q-p-s, "out of", q, "samples were analyzed.", p+s, "were corrupted or invalid length")
    print("number of Triggers:", r)
    print("run time:  ", datetime.now()-start)

