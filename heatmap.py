import matplotlib.cm as cm
import matplotlib.colors
from scipy.ndimage.filters import gaussian_filter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import *
import cartopy.crs as ccrs
from cartopy.io.img_tiles import *
import percache
"""
Percache is persistent caching. It will use up harddrive space to store outputs of functions.
You will have to delete the files occasionally:
my-cache.bak
my-cache.dat
my-cache.dir
"""



# remove na
coll = open_collisions()
coll = coll[~coll['X'].isna()]

#print(coll.groupby('LOCATION').sum())
#asdf
# remove outlier
coll = coll[coll['INJURIES'] < 50]

cmap = {
  'red': ((0.0, 0, 0), (0.35, 0, 0), (0.66, 1, 1), (0.89, 1, 1), (1, 0.5, 0.5)),
  'green': ((0.0, 0, 0), (0.125, 0, 0), (0.375, 1, 1), (0.64, 1, 1), (0.91, 0, 0), (1, 0, 0)),
  'blue': ((0.0, 0.5, 0.5), (0.11, 1, 1), (0.34, 1, 1), (0.65, 0, 0), (1, 0, 0)),
  #'alpha': ((0.0, 0.3, 0.3), (0.05, 0.3, 0.3), (0.3, .7, .7), (0.89, .9, .9), (1, 0.9, 0.9)),
 # 'alpha': ((0.0, 0.0, 0.0), (0.0, 1.0, 0.1), (1.0, 1.0, 1)),
  # (value left of anchor point, value right of anchor point, anchor point)
}
cm.myjet = matplotlib.colors.LinearSegmentedColormap(name='myjet', segmentdata=cmap, N=256, )#gamma=2.)

tiler = OSM()
geo = ccrs.PlateCarree()
x, y, z = np.hsplit(tiler.crs.transform_points(geo, coll.X.values, coll.Y.values), 3)
coll.loc[:,'X'] = x
coll.loc[:,'Y'] = y
xmi, xma, ymi, yma = coll.X.min(), coll.X.max(), coll.Y.min(), coll.Y.max()
xr, yr = xma - xmi, yma - ymi
map_extent = (xmi - xr*0.02, xma + xr*0.02, ymi - yr*0.02, yma + yr*0.02)
#data_extent = (xmi, xma, ymi, yma)

w = 4000
factor = (map_extent[3]-map_extent[2])/(map_extent[1]-map_extent[0])
h = int(w*factor)

@cache
def heatmap(x, y, sigma=1, weights=None, bins=5000):
    print('creating heatmap')
    hm, xedges, yedges = np.histogram2d(x, y, range=[[map_extent[0], map_extent[1]], [map_extent[2], map_extent[3]]], weights=weights, bins=bins)
    hm2 = gaussian_filter(hm, sigma=sigma)
    hm2/=(hm2.max()/hm.max())
    return hm2.T

def heatmap_postprocessing(img):
    return img#(img - img.min())/(img.max() - img.min())*256

def heatmap_plot(img, fname):
    print('saving heatmap', fname)
    fig, ax = plt.subplots(figsize=(w/1000, h/1000))
    ax = plt.axes(projection=tiler.crs)
    
    ax.set_extent(map_extent, crs=tiler.crs)
    
    composite = get_composite(tiler, map_extent)
    ax.imshow(composite[0], extent=composite[1], origin=composite[2])
    #aximg = ax.imshow(img, origin='lower', extent=data_extent, cmap=cm.myjet)#, alpha=.5)
    aximg = ax.imshow(img, origin='lower', extent=map_extent, cmap=cm.myjet, alpha=.5)
    fig.colorbar(aximg)
    plt.savefig(fname, dpi=1000)

@cache
def get_composite(tiler, extent):
    """
    This returns a background image. there might be more efficient ways of storing and rendering this quickly,
    but i am using percache for heatmap caching already.
    """
    z_target = 14
    composite = tiler.image_for_domain(sgeom.box(extent[0], extent[2], extent[1], extent[3]), target_z=z_target)
    return composite

img_col = heatmap(coll.X, coll.Y, weights=None, sigma=10.0, bins=[w, h])
#img_col = heatmap_postprocessing(img_col)
#heatmap_plot(img_col, 'heatmap-collisions.jpg')

img_inj = heatmap(coll.X, coll.Y, weights=coll.INJURIES, sigma=10.0, bins=[w, h])
#img_inj = heatmap_postprocessing(img_inj)
#heatmap_plot(img_inj, 'heatmap-injuries.jpg')

img_fat = heatmap(coll.X, coll.Y, weights=coll.FATALITIES, sigma=10.0, bins=[w, h])
#img_fat = heatmap_postprocessing(img_fat)
#heatmap_plot(img_fat, 'heatmap-fatalities.jpg')

img_inj_dens = np.where(img_col >= 1., img_inj*(img_col/img_col.sum()), 0)
#img_inj_dens = heatmap_postprocessing(img_inj_dens)
#heatmap_plot(img_inj_dens, 'heatmap-inj-dens.jpg')

img_fat_dens = np.where(img_col >= 1., img_fat*(img_col/img_col.sum()), 0)
#img_fat_dens = heatmap_postprocessing(img_fat_dens)
#heatmap_plot(img_fat_dens, 'heatmap-fat-dens.jpg')



# finds only rows contributing to intersections with 10 or more events
# significant locations
g = coll.groupby('LOCATION')
coll2 = coll[coll['LOCATION'].isin(g.count()['X'][g.count()['X']>10].index)]

img_col = heatmap(coll2.X, coll2.Y, weights=None, sigma=10.0, bins=[w, h])
#img_col = heatmap_postprocessing(img_col)
heatmap_plot(img_col, 'heatmap-collisions-sgnf.jpg')

img_inj = heatmap(coll2.X, coll2.Y, weights=coll2.INJURIES, sigma=10.0, bins=[w, h])
#img_inj = heatmap_postprocessing(img_inj)
heatmap_plot(img_inj, 'heatmap-injuries-sgnf.jpg')

img_fat = heatmap(coll2.X, coll2.Y, weights=coll2.FATALITIES, sigma=10.0, bins=[w, h])
#img_fat = heatmap_postprocessing(img_fat)
heatmap_plot(img_fat, 'heatmap-fatalities-sgnf.jpg')

img_inj_dens = np.where(img_col >= 1., img_inj*(img_col/img_col.sum()), 0)
#img_inj_dens = heatmap_postprocessing(img_inj_dens)
heatmap_plot(img_inj_dens, 'heatmap-inj-dens-sgnf.jpg')

img_fat_dens = np.where(img_col >= 1., img_fat*(img_col/img_col.sum()), 0)
#img_fat_dens = heatmap_postprocessing(img_fat_dens)
heatmap_plot(img_fat_dens, 'heatmap-fat-dens-sgnf.jpg')