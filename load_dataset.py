import pandas as pd
import numpy as np

def load_ccpp_stream(filepath: str):
    df = pd.read_excel(filepath, sheet_name='Sheet1', engine='openpyxl')
    
    # Features: AT, V, AP, RH (4 columns)
    X = df[['AT', 'V', 'AP', 'RH']].values.astype(np.float64)
    # Label: PE (net energy output)
    y = df['PE'].values.astype(np.float64)
    
    return X, y