import matplotlib.cm as cm
import matplotlib.colors
from scipy.ndimage.filters import gaussian_filter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from utils import *
import geopandas as gpd
from shapely.geometry import Point, Polygon

# remove na
coll = open_collisions()
coll = coll[~coll['X'].isna()]

# remove outlier
coll = coll[coll['INJURIES'] < 50]

cmap = {
  'red': ((0.0, 0, 0), (0.35, 0, 0), (0.66, 1, 1), (0.89, 1, 1), (1, 0.5, 0.5)),
  'green': ((0.0, 0, 0), (0.125, 0, 0), (0.375, 1, 1), (0.64, 1, 1), (0.91, 0, 0), (1, 0, 0)),
  'blue': ((0.0, 0.5, 0.5), (0.11, 1, 1), (0.34, 1, 1), (0.65, 0, 0), (1, 0, 0)),
  'alpha': ((0.0, 0.3, 0.3), (0.05, 0.3, 0.3), (0.3, .7, .7), (0.89, .9, .9), (1, 0.9, 0.9)),
}

def heatmap(x, y, sigma=1, weights=None, bins=5000):
    hm, xedges, yedges = np.histogram2d(x, y, weights=weights, density=True, bins=bins)
    hm = gaussian_filter(hm, sigma=sigma)
    # left, right, bottom, top
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return hm.T, extent

fig, ax = plt.subplots(figsize=(5, 10))
m = Basemap(projection='mill',
            resolution='c',
            llcrnrlon=coll.X.min(),
            llcrnrlat=coll.Y.min(),
            urcrnrlon=coll.X.max(),
            urcrnrlat=coll.Y.max(),
            epsg = 2285,
            )

m.arcgisimage(service='World_Street_Map', xpixels = 5000, verbose= False)

cm.myjet = matplotlib.colors.LinearSegmentedColormap(name='myjet', segmentdata=cmap, N=256, gamma=2.)

img, extent = heatmap(coll.X, coll.Y, weights=coll.INJURIES, sigma=10.0)
# noremalize to 0-256
img = (img - img.min())/(img.max() - img.min())*256

# run each coordinate through the map, then return to extent format
extent = [i for _l in m(extent[:2], extent[2:]) for i in _l]
ax.imshow(img, extent=extent, origin='lower', cmap=cm.myjet, )

plt.savefig('heatmap.jpg', dpi=800)
plt.show()
asdf