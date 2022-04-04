import numpy as np
import pandas as pd

class SanityCheck():
    def __init__(self):
        pass
    def wear_detect(self, raw_data, x=1,y=1,z=9.65, stdev=0.35):
        acc_std = np.linalg.norm(raw_data[['accx','accy','accz']], axis=1)[:-2].std()
        accx = raw_data['accx'].values[:-1].mean()
        accy = raw_data['accy'].values[:-1].mean()
        accz = raw_data['accz'].values[:-1].mean()
        not_worn = (acc_std < stdev) & (np.abs(accz) > z) & (np.abs(accx) < x) & (np.abs(accy) <y)
	   
        return not_worn

    
    def check_signal(self, raw_data, signal_duration = [2, 5]):
        watch_not_worn = self.wear_detect(raw_data)
        if watch_not_worn:
            raise ValueError("The watch is not worn")
        l = len(raw_data)
        res = []
        for sdur in signal_duration:
            res.append( 0.8*60*20*sdur <l<  1.2*60*20*sdur)
        
        if not any(res):
            raise ValueError('Sample not the right size')
    
#	    if l< (0.8*signal_duration*60*20):
#	        raise ValueError("Sample is too short")
#	    if l> (1.2*signal_duration*60*20):
#	        raise ValueError("Sample is too long")


if __name__ == '__main__':

    datapath = 'D:/UCI/Unite/Unite_RCT/Source Data/Raw Data 5 minutes/data_uniterct148-2021-01-29-14-49-46.csv'
    data = pd.read_csv(datapath, header=0, delimiter='\t')

    mod = SanityCheck()
    mod.check_signal(data, [5,2])