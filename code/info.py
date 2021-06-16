import os
from pathlib import Path

class Info:
    
    def __init__(self, args):
        
        self.DIR_CURR = os.getcwd()
        self.DIR_DATA = os.path.join(self.DIR_CURR, '..', 'data')
        self.DIR_RECORD = os.path.join(self.DIR_CURR, '..', 'record')
        
        self.DIR_PAIR = os.path.join(self.DIR_RECORD, f'{args.database}_{args.profile_id}_Pair_{args.pair_id}')
        Path(self.DIR_PAIR).mkdir(parents=True, exist_ok=True)
        
        self.ID_TRIAL = 1 + max([-1] + [int(name.split('_')[1]) for name in os.listdir(self.DIR_PAIR) if name.split('_')[0]=='Trial'])
        self.DIR_TRIAL = os.path.join(self.DIR_PAIR, f'Trial_{self.ID_TRIAL}')
        Path(self.DIR_TRIAL).mkdir(parents=True, exist_ok=True)
        
        self.FILE_STDOUT = os.path.join(self.DIR_TRIAL, f'stdout.txt')
        self.FILE_RESULT = os.path.join(self.DIR_TRIAL, f'result.json')
        self.FILE_MODEL = os.path.join(self.DIR_TRIAL, f'model.pkl')
        
        self.FILE_CONFIG = os.path.join(self.DIR_DATA, args.database, 'configs.pkl')
        self.FILE_PROFILE = os.path.join(self.DIR_DATA, args.database, 'profiles.pkl')
        self.FILE_PAIRS = os.path.join(self.DIR_DATA, 'train_test_pairs.json')
        
        self.DATABASE_HyperRec = 'HyperRec'
        self.DATABASE_LCBench = 'LCBench'

        self.RESULT_ARGS = 'Args'
        self.RESULT_QUERY_OBSERVATION = 'Query_Observation'
        self.RESULT_MAX_RANKING = 'Max_Ranking'

        self.DEVICE = 'cuda'
        self.SEED = 2021
        self.EXTREME_SMALL = -1e10