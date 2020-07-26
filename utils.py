import os.path
import pandas as pd
import pickle
import geopandas as gpd

def pickleable(func):

  def inner(*args, **kwargs):
    pfile = kwargs['file']+str('.pickle')
    if os.path.exists(pfile):
      with open(pfile, 'rb') as f:
        return pickle.load(f)
    df = func(*args, **kwargs)
    with open(pfile, 'wb') as f:
      pickle.dump(df, f)
    return df
  return inner
  

@pickleable
def _open_collisions(file):
  return pd.read_csv(file=file, parse_dates=['INCDATE', 'INCDTTM'])
def open_collisions():
  return _open_collisions(file='./data/Collisions.csv')
  

@pickleable
def _open_streets(file):
 return gpd.read_file(file)

def open_streets():
 return _open_streets(file="./data/Seattle_Streets-shp/Seattle_Streets.shp")