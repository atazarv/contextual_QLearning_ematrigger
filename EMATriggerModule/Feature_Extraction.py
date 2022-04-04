import numpy as np
import pandas as  pd
import heartpy as hp

class PPGFeatureExtraction():
    def __init__(self, raw_data):
        self.data = raw_data

    def PPG_features(self, data):
        raw = data.ppg.values
        timer = data.timestamp.values
        sample_rate = hp.get_samplerate_mstimer(timer)
        filtered = hp.filter_signal(raw, [0.7, 3.5], sample_rate=sample_rate, order=3, filtertype='bandpass')
        wd, m = hp.process(filtered, sample_rate = sample_rate, clean_rr=True, calc_freq=False)

        return m

    def FeatureExtract(self):
        ppg_features_dict = self.PPG_features(self.data[['ppg', 'timestamp']])
        ppg_features = list(ppg_features_dict.values())
        if (ppg_features.count(np.nan) or (ppg_features.count(0)>2) or ((np.sum(ppg_features)/(10e10))>1)):
            print('Error extracting feature.')
            print('Feature list', (ppg_features))

            raise ValueError("Returned error while extracting features")

        return list(ppg_features_dict.keys()), ppg_features

if __name__ == '__main__':

    datapath = 'D:/UCI/Unite/Unite_RCT/Source Data/Raw Data 5 minutes/data_uniterct148-2021-01-29-14-49-46.csv'
    data = pd.read_csv(datapath, header=0, delimiter='\t')

    mod = PPGFeatureExtraction(data)
    print(mod.FeatureExtract())