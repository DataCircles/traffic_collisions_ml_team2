import os.path
import pandas as pd

import percache
cache = percache.Cache("my-cache")

@cache
def open_collisions(file='./data/Collisions.csv'):
  return pd.read_csv(file, parse_dates=['INCDATE', 'INCDTTM'])
