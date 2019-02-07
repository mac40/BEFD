import numpy as np
import pandas as pd


df = pd.read_excel('USINDPROD_M_NOV_2018.xls', skiprows=10,index_col='observation_date',columns=['observation_date','IPB50001N'])

new_df = np.array(df['IPB50001N'])


np.savetxt('usindprod.csv',new_df,delimiter=",")