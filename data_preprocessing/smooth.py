import numpy as np
import pandas as pd


def smooth_exp(cnt: pd.DataFrame):
    """Apply smoothing to gene expression data in Pandas DataFrame.
    Take average gene expression of the nearest 9 spots.
    
    Args:
        cnt (pd.DataFrame): count data 

    Returns:
        pd.DataFrame: smoothed expression in DataFrame. 
    """

    ids = cnt.index.str.split('x').map(lambda x: f"{int(x[0])}x{int(x[1])}")

    # the 8-neighborhood offsets plus the center spot itself
    delta = np.array([[1,0],
            [0,1],
            [-1,0],
            [0,-1],
            [1,1],
            [-1,-1],
            [1,-1],
            [-1,1],
            [0,0]])

    cnt_smooth = np.zeros_like(cnt).astype('float')

    for i in range(len(cnt)):
        spot = cnt.iloc[i,:]
        center = np.array(spot.name.split('x')).astype('int')
        neighbors = center - delta
        neighbors = pd.DataFrame(neighbors).astype('str').apply(lambda x: "x".join(x), 1)
        cnt_smooth[i,:] = cnt[ids.isin(neighbors)].mean(0)
        
    cnt_smooth = pd.DataFrame(cnt_smooth)
    cnt_smooth.columns = cnt.columns
    cnt_smooth.index = cnt.index
    
    return cnt_smooth