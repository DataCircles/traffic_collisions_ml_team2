
import numpy as np

try:
  profile
except:
  def profile(ob):
    return ob

"""
cramer's v is symmetric nominal association.
"""
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x,y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))

def conditional_entropy(x, y, ic=None, jic=None):
    # x and y should be 1-d arrays
    # H(x|y)
    i, c = ic if ic else np.unique(y, return_counts=True, return_inverse=True)[1:]
    i2, c2 = np.unique(y, return_counts=True, return_inverse=True)[1:]
    #joint
    ji, jc = jic if jic else np.unique(np.dstack([x, y]), return_counts=True, return_inverse=True, axis=1)[1:]
    entropy = -(np.log(jc[ji])-np.log(c[i])).sum()/len(x)
    return entropy

def entropy(x, ic=None):
    # x should be a 1-d array
    (i, c) = ic if ic else np.unique(x, return_counts=True, return_inverse=True)[1:]
    ent = -(np.log(c[i])-np.log(len(x))).sum()/len(x)
    return ent

def uncertainty(x, y):
    # x and y should be 1-d arrays
    # U(X|Y)
    s_xy = conditional_entropy(x, y)
    ent = entropy(x)
    res = (ent - s_xy) / ent if ent else 1
    return res
    
def entropy2d(x, ic=None):
    # x should be a 2d array where features are columns
    ic = ic if ic else [np.unique(x, return_counts=True, return_inverse=True)[1:] for x in x.T]
    ent = [-(np.log(c[i])-np.log(len(x))).sum()/len(x) for i, c in ic]
    return ent

def uncertainty2D(arr):
    # arr should be a 2d array where features are columns
    cols = arr.shape[1]
    ic = [np.unique(x, return_counts=True, return_inverse=True)[1:] for x in arr.T]
    jic = {}
    for x in range(cols):
      for y in range(x+1):
        jic[(x, y)] = np.unique(arr.T[[x, y]], return_counts=True, return_inverse=True, axis=1)[1:]
    ent = entropy2d(arr, ic=ic)
    def f(x, y):
      return ((ent[x] - conditional_entropy(arr.T[x], arr.T[y], ic=ic[y], jic=jic[(max(x, y), min(x, y))]))/ent[x]) if ent[x] else 1
    res = np.array([f(x, y) for y in range(cols) for x in range(cols)]).reshape(cols, cols)
    return res
    
def correlation_ratio(categories, measurements):
    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat)+1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0,cat_num):
        cat_measures = measurements[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array,n_array))/np.sum(n_array)
    numerator = np.sum(np.multiply(n_array,np.power(np.subtract(y_avg_array,y_total_avg),2)))
    denominator = np.sum(np.power(np.subtract(measurements,y_total_avg),2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = np.sqrt(numerator/denominator)
    return eta