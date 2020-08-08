import heatmap as hm
from utils import *

# remove na
coll = collision_street_intersection_joined()

coll = coll[~coll['x'].isna()]
coll = coll[coll['collisiontype'].notna()]

heat = hm.HeatMap(coll)


cyc = coll[coll['colltype3']=='Cyc']
img_cyc = heat.heatmap(cyc.x, cyc.y, weights=None, sigma=10.0)
#heat.heatmap_plot(img_cyc, 'heatmap-cycl.jpg')


ped = coll[coll['colltype3']=='Ped']
img_ped = heat.heatmap(ped.x, ped.y, weights=None, sigma=10.0)
#heat.heatmap_plot(img_ped, 'heatmap-ped.jpg')


g = ped.set_index('incdttm').groupby(pd.Grouper(freq='Y'))
keys = g.groups
for i, k, in enumerate(keys):
  print(i, k)
  img = heat.heatmap(*(g.get_group(k)[['x', 'y']].values.T), weights=None, sigma=10.0)
  heat.heatmap_plot(img, 'heatmap-ped-yearly-'+str(i)+'.jpg')


# ffmpeg -framerate 1 -f image2 -i heatmap-ped-yearly-%1d.jpg heatmap-ped.mp4