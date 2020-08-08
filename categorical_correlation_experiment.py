import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss
from correlation import *

from utils import *

set_4k_display_settings()
df = collision_street_intersection_joined()

def plot(corr, cols):
    fig = plt.figure(figsize=(13,13))
    ax = fig.add_subplot(111)
    cax = ax.matshow(corr, cmap="inferno", vmin=0, vmax=1)
    
    for (i, j), z in np.ndenumerate(corr):
        ax.text(j, i, '{:0.2f}'.format(z), color='k' if z > 0.5 else 'w', ha='center', va='center')
    
    fig.colorbar(cax)
    ticks = np.arange(0, len(cols), 1)
    ax.set_xticks(ticks)
    plt.xticks(rotation=90)
    ax.set_yticks(ticks)
    ax.set_xticklabels(cols, rotation=90, horizontalalignment="center")
    ax.set_yticklabels(cols)
    plt.show();


#print(df2[cols].apply(pd.Series.value_counts))

"""
how all the data looks
"""    
#df2 = df.copy()
#cat_columns = df2.select_dtypes(['category']).columns
#df2[cat_columns] = df2[cat_columns].apply(lambda x: x.cat.codes)
#Combine cat coded cols and binary cols to creat a short list of categories we might be interested in.
#cols = df2.select_dtypes(np.int8).columns.tolist() + df2.select_dtypes(np.bool).columns.tolist()
#df2[cols] = df2[cols].apply(lambda x: x.astype(np.int8))
#corr = uncertainty2D(df2[cols].values)
#plot(corr, cols)


"""
look at only high sev collisions
"""
#df2 = df.copy()
#df2 = df2[(df2['severitycode']=='2')|(df2['severitycode']=='2b')|(df2['severitycode']=='3')|(df2['severitycode']=='3')]
#cat_columns = df2.select_dtypes(['category']).columns
#for cc in cat_columns:
#    df2[cc].cat.remove_unused_categories(inplace=True)
#df2[cat_columns] = df2[cat_columns].apply(lambda x: x.cat.codes)
#cols = df2.select_dtypes(np.int8).columns.tolist() + df2.select_dtypes(np.bool).columns.tolist()
#df2[cols] = df2[cols].apply(lambda x: x.astype(np.int8))
#corr = uncertainty2D(df2[cols].values)
#plot(corr, cols)


"""
split data by all uncontrollables.
"""
df2 = df.copy()


#g = df2.groupby(['timeofday', 'weather', 'roadcond', 'lightcond', 'colltype3'], observed=True)
#g.sum()

df2 = df2[df2['colltype3'] == 'Cyc']
print(df2.head(100))
cat_columns = df2.select_dtypes(['category']).columns
for cc in cat_columns:
    df2[cc].cat.remove_unused_categories(inplace=True)
df2[cat_columns] = df2[cat_columns].apply(lambda x: x.cat.codes)
cols = df2.select_dtypes(np.int8).columns.tolist() + df2.select_dtypes(np.bool).columns.tolist()
df2[cols] = df2[cols].apply(lambda x: x.astype(np.int8))
corr = uncertainty2D(df2[cols].values)
plot(corr, cols)





