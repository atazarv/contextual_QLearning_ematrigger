import pandas as pd

class LabelMatching():
    def __init__(self):
        pass
    def match(self, data, sr_labels, header, temporal_threshhold=32*6e4, 
              stress_labels = ['not at all', 'a little bit', 'some', 'a lot', 'extremely']):
        labels = sr_labels.copy()
        unlabeled_data = data.copy()
        #types of valid stress labels:
        #stress_labels = ['not at all', 'a little bit', 'some', 'a lot', 'extremely'] 
        #stress_labels = [1,2,3,4,5]
        labels = labels[labels['data.stressed']!='undefined']
        labels = labels[labels['data.stressed'].isin(stress_labels)]
        labels = labels.drop_duplicates()

        labels.rename(columns={"data.stressed_last_modify": "timestamp_label", 'name': 'user', 'data.stressed':'reported_stress'}, inplace=True)
        labels = labels[['timestamp_label', 'user', 'reported_stress']]
        labels['timestamp_label'] = pd.to_numeric(labels['timestamp_label']) 
        labels.sort_values(by='timestamp_label', inplace=True, ignore_index=True)
        labels['timestamp_label'] = labels.timestamp_label.astype('int64')

        unlabeled_data['timestamp1_sample'] = unlabeled_data.timestamp1_sample.astype('int64')
        unlabeled_data['timestamp2_sample'] = unlabeled_data.timestamp2_sample.astype('int64')
        unlabeled_data.sort_values(by='timestamp2_sample', inplace=True, ignore_index=True)
        unlabeled_data.drop(columns = ['timestamp_label', 'resp_t_min', 'reported_stress'], inplace=True)

        labeled_data   = pd.merge_asof(labels, unlabeled_data.reset_index(), left_on = 'timestamp_label' , right_on = 'timestamp2_sample', by='user', direction='backward')
        #if we match more than one label to one sample, we keep the 'first' label only
        labeled_data.drop_duplicates(subset = ['index', 'user'], keep = 'first', inplace=True)
        labeled_data = labeled_data[labeled_data['index'].notnull()]

        labeled_data_c = unlabeled_data.reset_index().merge(labeled_data[['index', 'user', 'timestamp_label', 'reported_stress']], how = 'left', on = ['index', 'user'])

        triggered_data = unlabeled_data[unlabeled_data['triggered']==1] #& (unlabeleed_data['realtime']==1)
        triggered_data = triggered_data[['timestamp1_sample', 'timestamp2_sample', 'user']]
         
        triggered_data = pd.merge_asof(triggered_data, labels, left_on = 'timestamp2_sample', right_on= 'timestamp_label', by = 'user', direction='forward')
        triggered_data['resp_t_min'] = (triggered_data['timestamp_label'] - triggered_data['timestamp2_sample'])/(6e4)
        triggered_data = triggered_data[['timestamp1_sample', 'timestamp2_sample', 'user', 'resp_t_min']]

        labeled_data_final = labeled_data_c.merge(triggered_data, how = 'left', on = ['timestamp1_sample', 'timestamp2_sample', 'user'])

        return labeled_data_final[header]