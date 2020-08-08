import matplotlib.cm as cm
import matplotlib.colors
from scipy.ndimage.filters import gaussian_filter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import *
import cartopy.crs as ccrs
from cartopy.io.img_tiles import *
import matplotlib.animation as anim


import percache
"""
Percache is persistent caching. It will use up harddrive space to store outputs of functions.
You will have to delete the files occasionally:
my-cache.bak
my-cache.dat
my-cache.dir
"""

@cache
def get_composite(tiler, extent):
    """
    This returns a background image. there might be more efficient ways of storing and rendering this quickly,
    but i am using percache for heatmap caching already.
    """
    z_target = 14
    composite = tiler.image_for_domain(sgeom.box(extent[0], extent[2], extent[1], extent[3]), target_z=z_target)
    return composite


@cache
def _heatmap(x, y, w, h, map_extent, sigma=1, weights=None, bins=None):
    bins = bins if bins else [w, h]
    hm, xedges, yedges = np.histogram2d(x, y, range=[[map_extent[0], map_extent[1]], [map_extent[2], map_extent[3]]], weights=weights, bins=bins)
    hm2 = gaussian_filter(hm, sigma=sigma)
    hm2/=(hm2.max()/hm.max())
    return hm2.T


class HeatMap():
  
  cmap = {
    'red': ((0.0, 0, 0), (0.35, 0, 0), (0.66, 1, 1), (0.89, 1, 1), (1, 0.5, 0.5)),
    'green': ((0.0, 0, 0), (0.125, 0, 0), (0.375, 1, 1), (0.64, 1, 1), (0.91, 0, 0), (1, 0, 0)),
    'blue': ((0.0, 0.5, 0.5), (0.11, 1, 1), (0.34, 1, 1), (0.65, 0, 0), (1, 0, 0)),
    #'alpha': ((0.0, 0.3, 0.3), (0.05, 0.3, 0.3), (0.3, .7, .7), (0.89, .9, .9), (1, 0.9, 0.9)),
   # 'alpha': ((0.0, 0.0, 0.0), (0.0, 1.0, 0.1), (1.0, 1.0, 1)),
    # (value left of anchor point, value right of anchor point, anchor point)
  }
  jet = matplotlib.colors.LinearSegmentedColormap(name='myjet', segmentdata=cmap, N=256)#gamma=2.)
  
  def __init__(self, df, xname='x', yname='y', width=4000):
    self.xname = xname
    self.yname = yname
    
    self.tiler = OSM()
    self.geo = ccrs.PlateCarree()
    self.convert_xy(df)
  
    xmi, xma, ymi, yma = df[xname].min(), df[xname].max(), df[yname].min(), df[yname].max()
    xr, yr = xma - xmi, yma - ymi
    self.map_extent = (xmi - xr*0.02, xma + xr*0.02, ymi - yr*0.02, yma + yr*0.02)
    #data_extent = (xmi, xma, ymi, yma)
    self.w = width
    factor = (self.map_extent[3]-self.map_extent[2])/(self.map_extent[1]-self.map_extent[0])
    self.h = int(self.w*factor)
  
  def convert_xy(self, df):
    x, y, z = np.hsplit(self.tiler.crs.transform_points(self.geo, df[self.xname].values, df[self.yname].values), 3)
    print('Warning: converting x and y values to heatmap extent')
    df.loc[:, self.xname] = x
    df.loc[:, self.yname] = y
  
  def heatmap(self, x, y, sigma=1, weights=None, bins=None):
    
    return _heatmap(x, y, self.w, self.h, self.map_extent, sigma=sigma, weights=weights, bins=bins)
  
  def heatmap_postprocessing(self, img):
    return img#(img - img.min())/(img.max() - img.min())*256
  
  def heatmap_plot(self, img, fname):
    print('saving heatmap', fname)
    fig, ax = plt.subplots(figsize=(self.w/1000, self.h/1000))
    ax = plt.axes(projection=self.tiler.crs)
    
    ax.set_extent(self.map_extent, crs=self.tiler.crs)
    
    composite = get_composite(self.tiler, self.map_extent)
    ax.imshow(composite[0], extent=composite[1], origin=composite[2])
    #aximg = ax.imshow(img, origin='lower', extent=data_extent, cmap=cm.myjet)#, alpha=.5)
    aximg = ax.imshow(img, origin='lower', extent=self.map_extent, cmap=HeatMap.jet, alpha=.5)
    fig.colorbar(aximg)
    plt.savefig(fname, dpi=1000)
  
  #def heatmap_mov(self, imgs, fname):
  #  # Set up formatting for the movie files
  #  Writer = anim.writers['ffmpeg']
  #  writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
  #  fig2 = plt.figure()
  #  im_ani = anim.ArtistAnimation(fig2, imgs, interval=50, repeat_delay=3000, blit=True)
  #  im_ani.save('im.mp4', writer=writer)

if __name__=='__main__':
  coll = open_collisions()
  hm = HeatMap(coll, xname='X', yname='Y')
  
  
  img_col = hm.heatmap(coll.Y, coll.Y, weights=None, sigma=10.0)
  #img_col = heatmap_postprocessing(img_col)
  #heatmap_plot(img_col, 'heatmap-collisions.jpg')

  img_inj = hm.heatmap(coll.X, coll.Y, weights=coll.INJURIES, sigma=10.0)
  #img_inj = heatmap_postprocessing(img_inj)
  #heatmap_plot(img_inj, 'heatmap-injuries.jpg')

  img_fat = hm.heatmap(coll.X, coll.Y, weights=coll.FATALITIES, sigma=10.0)
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

  img_col = hm.heatmap(coll2.X, coll2.Y, weights=None, sigma=10.0)
  #img_col = heatmap_postprocessing(img_col)
  hm.heatmap_plot(img_col, 'heatmap-collisions-sgnf.jpg')

  img_inj = hm.heatmap(coll2.X, coll2.Y, weights=coll2.INJURIES, sigma=10.0)
  #img_inj = heatmap_postprocessing(img_inj)
  hm.heatmap_plot(img_inj, 'heatmap-injuries-sgnf.jpg')

  img_fat = hm.heatmap(coll2.X, coll2.Y, weights=coll2.FATALITIES, sigma=10.0)
  #img_fat = heatmap_postprocessing(img_fat)
  hm.heatmap_plot(img_fat, 'heatmap-fatalities-sgnf.jpg')

  img_inj_dens = np.where(img_col >= 1., img_inj*(img_col/img_col.sum()), 0)
  #img_inj_dens = heatmap_postprocessing(img_inj_dens)
  hm.heatmap_plot(img_inj_dens, 'heatmap-inj-dens-sgnf.jpg')

  img_fat_dens = np.where(img_col >= 1., img_fat*(img_col/img_col.sum()), 0)
  #img_fat_dens = heatmap_postprocessing(img_fat_dens)
  hm.heatmap_plot(img_fat_dens, 'heatmap-fat-dens-sgnf.jpg')