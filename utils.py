import os.path
import pandas as pd
import numpy as np
import percache
import re

cache = percache.Cache("my-cache")

@cache
def open_collisions(file='./data/Collisions.csv'):
  return pd.read_csv(file, parse_dates=['INCDATE', 'INCDTTM'], dtype={'sdot_colcode':'category'})



"""

data from 2003 months nov and dec are missing, there is one datapoint in october 2003.



sdotcolnum   all values from april 2013 onward are missing except for month september 2017
seglanekey   all values from 2004 to 2006 are 0
crosswalkkey all values from 2004 to 2006 are 0
"""


def set_4k_display_settings():
  pd.set_option('display.max_colwidth', None)
  pd.set_option('display.max_columns', None)
  pd.set_option('display.width', None)
  pd.set_option('display.max_rows', 100)
  pd.set_option('display.min_rows', 100)


num_cols = ['personcount', 'pedcount', 'pedcylcount', 'vehcount', 'injuries', 'seriousinjuries', 'fatalities']
cat_cols = ['status', 'addrtype', 'exceptrsncode', 'severitycode', 'collisiontype', 'junctiontype', 'sdot_colcode', 'inattentionind', 'underinfl', 'weather', 'roadcond', 'lightcond',  'pedrownotgrnt', 'speeding', 'st_colcode', 'hitparkedcar', 'reporttype']


@cache
def cleaned_collisions():
  # remove na
  odf = open_collisions()
  odf.columns = map(str.lower, odf.columns)



  """
  Clean the data.
  Clean the categorical data.
  """
  df = odf.copy()

  
  
  df.drop('objectid', inplace=True, axis = 1)

  df.drop('inckey', inplace=True, axis = 1)

  df.drop('coldetkey', inplace=True, axis = 1)

  # truncate report number, first part of report string might be useful

  df['reporttype'] = np.where( df['reportno'].str.startswith('EA'), 'EA', np.where(df['reportno'].str.startswith('E'), 'E', np.where(df['reportno'].str.startswith('C'), 'C', 'N')))
  df.drop('reportno', inplace=True, axis = 1)


  # status seems to correspond to whether an st_colcode exists, maybe there are two systems and the values were merged. if it is matched, then collisiontype, lighcond, roadcond, and weather are present.
  df['status'] = df['status'].astype('category')

  #df['addrtype'].fillna(value='N/A', inplace=True)
  df['addrtype'] = df['addrtype'].astype('category')

  #df.drop('intkey', inplace=True, axis = 1)

  """
  This feature sometimes imports a string ' ' other times NaN, so unify those.
  """
  loc = df['exceptrsncode'] == 'NEI'
  df.loc[loc, 'exceptrsncode'] = 1
  df.loc[~loc, 'exceptrsncode'] = 0
  df['exceptrsncode'] = df['exceptrsncode'].astype(np.bool)

  df.drop('exceptrsndesc', inplace=True, axis = 1)

  #df['severitycode'].fillna(value='N/A', inplace=True)
  df['severitycode'] = df['severitycode'].astype('category')

  df.drop('severitydesc', inplace=True, axis = 1)

  #df['collisiontype'].fillna(value='N/A', inplace=True)
  df['collisiontype'] = df['collisiontype'].astype('category')

  # np.all(df['incdttm'].dt.to_period('d') == df['incdate'].dt.to_period('d')) == True, seems safe to drop incdate
  df.drop('incdate', inplace=True, axis = 1)

  #df['junctiontype'].fillna(value='N/A', inplace=True)
  df['junctiontype'] = df['junctiontype'].astype('category')

  df.loc[df['sdot_colcode']==0.0, 'sdot_colcode'] = np.nan
  #df['sdot_colcode'].fillna(value='N/A', inplace=True)
  df['sdot_colcode'] = df['sdot_colcode'].astype('category')



  df.drop('sdot_coldesc', inplace=True, axis = 1)


  loc = ~df['inattentionind'].isna()
  df.loc[loc, 'inattentionind'] = 1
  df['inattentionind'].fillna(value=0, inplace=True)
  df['inattentionind'] = df['inattentionind'].astype(np.bool)


  """
  underinfl switched from 0/1 to Y/N over time
  df2 = df.set_index('incdttm')
  g = df2['underinfl'].isna().groupby(pd.Grouper(freq='Y'))
  g.value_counts()
  """
  loc = np.logical_or(df['underinfl'] == 'Y',df['underinfl'] == '1')
  df.loc[loc, 'underinfl'] = 1
  loc = np.logical_or(df['underinfl'] == 'N', df['underinfl'] == '0')
  df.loc[loc, 'underinfl'] = 0
  df['underinfl'] = df['underinfl'].astype(np.bool)



  #df['weather'].fillna(value='N/A', inplace=True)
  #df = df[~df['weather'].isna()]
  df['weather'] = df['weather'].astype('category')

  #df['roadcond'].fillna(value='N/A', inplace=True)
  #df = df[~df['roadcond'].isna()]
  df['roadcond'] = df['roadcond'].astype('category')

  #df['lightcond'].fillna(value='N/A', inplace=True)
  df['lightcond'] = df['lightcond'].astype('category')


  loc = ~df['pedrownotgrnt'].isna()
  df.loc[loc, 'pedrownotgrnt'] = 1
  df['pedrownotgrnt'].fillna(value=0, inplace=True)
  df['pedrownotgrnt'] = df['pedrownotgrnt'].astype(np.bool)


  """
  sdotcolnum stopped being applied over time
  df2 = df.set_index('incdttm')
  g = df2['sdotcolnum'].isna().groupby(pd.Grouper(freq='Y'))
  g.value_counts()
  """
  df.drop('sdotcolnum', inplace=True, axis = 1)



  loc = ~df['speeding'].isna()
  df.loc[loc, 'speeding'] = 1
  df['speeding'].fillna(value=0, inplace=True)
  df['speeding'] = df['speeding'].astype(np.bool)


  df.loc[df['st_colcode']==' ', 'st_colcode'] = np.nan
  #df['st_colcode'].fillna(value='N/A', inplace=True)
  df['st_colcode'] = df['st_colcode'].astype('category')


  df.drop('st_coldesc', inplace=True, axis = 1)

  """
  If we knew more about the lane key like northbound/southbound, etc., it would be more useful.
  At least record whether the key is present.
  """
  loc = df['seglanekey'] > 0
  df.loc[~loc, 'seglanekey'] = 0
  df.loc[loc, 'seglanekey'] = 1
  df['seglanekey'] = df['seglanekey'].astype(np.bool)

  """
  Knowing if a crosswalk was involved is interesting.
  """
  loc = df['crosswalkkey'] > 0
  df.loc[~loc, 'crosswalkkey'] = 0
  df.loc[loc, 'crosswalkkey'] = 1
  df['crosswalkkey'] = df['crosswalkkey'].astype(np.bool)

  loc = df['hitparkedcar'] == 'N'
  df.loc[loc, 'hitparkedcar'] = 0
  df.loc[~loc, 'hitparkedcar'] = 1
  df['hitparkedcar'] = df['hitparkedcar'].astype(np.bool)

  """Cleaning Section Done"""

  """
  Now add some useful feature columns
  """
  
  
  df['colltype3'] = df['collisiontype'].apply(lambda x: 'Ped' if x == 'Pedestrian' else 'Cyc' if x == 'Cycles' else 'Veh')
  df['colltype3'] = df['colltype3'].astype('category')

  b = [0,4,8,12,16,20,24]
  l = ['Late Night', 'Early Morning','Morning','Noon','Eve','Night']
  df['timeofday'] = pd.cut(df['incdttm'].dt.hour, bins=b, labels=l, include_lowest=True)
  df['timeofday'] = df['timeofday'].astype('category')

  return df











#df2 = (df[(df['severitycode']=='0') | (df['severitycode']=='1')]).copy()
#df2['severitycode'].cat.remove_unused_categories(inplace=True)
#cat = 'severitycode'; print(df2[cat].cat.categories); print(reduce(lambda x, y: x+','+y, [df2[df2[cat]==x][cat_cols].apply.astype('str') for x in df2[cat].cat.categories]))

  

##df2.set_index('incdttm', inplace=True)
#df2.groupby(pd.Grouper(freq='Y')).sum()

#df2['month'] = df2['incdttm'].dt.to_period('M')
#df2.groupby(['severitycode', 'month']).sum()

"""
# so lets reverse engineer severitycode

#df.loc[(df['fatalities'] < 1) & (df['seriousinjuries'] >= 1), 'severitycode'] = '2b'
#df.loc[(df['fatalities'] < 1) & (df['seriousinjuries'] >= 1), 'severitycode'] = '2b'




df.drop('incdate', inplace=True, axis = 1)


df['sevcode2'] = np.where(df['fatalities'] > 0, '3',
                            np.where(
                              df['injuries'] > 0, 
                              np.where(df['seriousinjuries'] > 0, '2b', '2'), 
                              np.where(
                                df['status'] == 'Matched',
                                np.where(
                                  (df['vehcount'] > 0)
                                  | (df['personcount']>0)
                                  #| (df['sdot_colcode'] > 0)
                                  | (~df['st_colcode'].isna()) 
                                  ,'1m'
                                  ,'0m'
                                  ),
                                np.where(
                                  df['exceptrsncode']==False,
                                  '1u', 
                                  np.where(
                                    (df['personcount'] > 0),
                                    '1u2',
                                    '0u2',
                                    )
                                  )
                                )
                              )
                            )
                              
                              
#df['sevcode2'] = np.where(df['fatalities'] > 0, '3', np.where(df['injuries'] > 0, np.where(df['seriousinjuries'] > 0, np.where(df['vehcount'] > 0, '1', '?')))
pd.crosstab(df['sevcode2'], df['severitycode'], dropna=False)
df[(df['severitycode'] == '0')]
df2 = df[(df['fatalities'] == 0) & (df['injuries'] > 0) & (~(df['seriousinjuries']>1))]



p = 'Q'; df2[p] = df2['incdttm'].dt.to_period(p); pd.concat([df2[['severitycode', p, 'lightcond']], pd.get_dummies(df2['lightcond'], dummy_na=True)], axis=1).groupby(['severitycode', p]).sum()
c = 'sdot_colcode'; p = 'Q'; df2[p] = df2['incdttm'].dt.to_period(p); pd.concat([df2[['severitycode', p, c]], pd.get_dummies(df2[c], dummy_na=True)], axis=1).groupby([p]).sum()





cat = 'severitycode'; print(reduce(lambda x, y: x+','+y, [df[df[cat]==x][cat_cols].apply(pd.Series.value_counts).astype('str') for x in df[cat].cat.categories]))


#for nc in num_cols:
#  df['has'+nc] = ~df[nc].isna()
#  df['has'+nc].value_counts()
#  cat_cols.append('has'+nc)

df['status'].unique()
print(df[df['status']=='Matched'][cat_cols].apply(pd.Series.value_counts))
print(df[df['status']=='Unmatched'][cat_cols].apply(pd.Series.value_counts))

"""

@cache
def read_streets():
  df = pd.read_csv('./data/Seattle_Streets.csv')
  df.columns = map(str.lower, df.columns)
  return df
  


@cache
def read_intersections():
  df = pd.read_csv('./data/Intersections.csv')
  df.columns = map(str.lower, df.columns)
  return df

@cache
def build_intersection_speed_map(s):
  ismap = {}
  for idx, row in s[~s['s_unitdesc'].isna()].iterrows():
    for intkey in (row['s_intrlo'], row['s_intrhi']):
      if intkey not in ismap:
        ismap[intkey] = set()
      ismap[intkey].add(row['s_speedlimit'])
  return ismap
 

@cache
def collision_street_intersection_joined():
  s = read_streets()
  s.columns = ['s_'+c for c in s.columns]  
  ismap = build_intersection_speed_map(s)  
  i = read_intersections()
  i.columns = ['i_'+c for c in i.columns]
  c = cleaned_collisions()
  #c.loc[:, 'location'] = c['location'].str.strip()
  #s.loc[:, 's_unitdesc'] = s['s_unitdesc'].str.strip()
  #i.loc[:, 'i_unitdesc'] = i['i_unitdesc'].str.strip()
  cs = c.join(s.set_index('s_unitdesc'), on='location', how='left')
  csi = cs.join(i.set_index('i_unitdesc'), on='location', how='left')
  # set speed limit hi and lo variables for intersection or block
  csi['speedlimitlo'] = np.nan
  csi['speedlimithi'] = np.nan
  l = csi['addrtype']=='Intersection'
  csi.loc[l, 'speedlimitlo'] = csi[l]['location'].apply(lambda x: min(ismap[x]) if x in ismap else np.nan)
  csi.loc[l, 'speedlimithi'] = csi[l]['location'].apply(lambda x: max(ismap[x]) if x in ismap else np.nan)
  l = csi['addrtype']=='Block'
  csi.loc[l, 'speedlimitlo'] = csi[l]['s_speedlimit']
  csi.loc[l, 'speedlimithi'] = csi[l]['s_speedlimit']
  return csi
  
"""
#
g = cs.groupby('location')
cs2 = cs[cs['location'].isin(g.count()['x'][g.count()['x']>10].index)]

(cs['speedlimithi']-cs['speedlimitlo']).value_counts()/(~cs['speedlimithi'].isna()).sum()
(cs2['speedlimithi']-cs2['speedlimitlo']).value_counts()/(~cs2['speedlimithi'].isna()).sum()


cs[(cs['addrtype']=='Intersection')]['location']
cs[(cs['addrtype']=='Intersection') & cs['speedlimithi'].isna()]['location']

"""


"""

f = pd.read_csv('./data/Traffic_Flow_Map_Volumes.csv')
h0, h1 = np.histogram(f[f['YEAR']==2017]['AAWDT']); h0 = h0/h0.sum()
h0, h1 = np.histogram(f[f['YEAR']==2018]['AAWDT']); h0 = h0/h0.sum()
"""

