import pandas as pd
from config.constants import *
from config.config import *
from config.utils import *

logger.success('Modules imported!')
logger.info(f'The solution is {SOLUTION}')

data = pd.read_pickle('modeling/src/data/main_data.pkl')
preprocessed_data = preprocessing(data.head())
print(preprocessed_data)