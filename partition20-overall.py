
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from __future__ import print_function
from sklearn.cluster import KMeans


# In[24]:

get_ipython().magic('pylab inline')
import pandas as pd, numpy as np             # Data manipulation 
from sklearn.decomposition import PCA        # The main algorithm
from matplotlib import pyplot as plt         # Graphing
import seaborn as sns                        # Graphing
from collections import defaultdict, Counter # Utils
sns.set(style="white")                       # Tuning the style of charts
import warnings                              # Disable some warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
get_ipython().magic('matplotlib inline')


# In[3]:

partition20=pd.read_csv('/Users/huyuxuan/desktop/20.csv')


# In[7]:

# Take a handle on the dataset
mydataset = partition20

# Load the first lines.
# You can also load random samples, limit yourself to some columns, or only load
# data matching some filters.
#
# Please refer to the Dataiku Python API documentation for more information
df = mydataset

df_orig = df.copy()

# Get the column names
numerical_columns = list(df.select_dtypes(include=[np.number]).columns)
categorical_columns = list(df.select_dtypes(include=[object]).columns)
date_columns = list(df.select_dtypes(include=['<M8[ns]']).columns)

# Print a quick summary of what we just loaded
print ("Loaded dataset")
print ("   Rows: %s" % df.shape[0])
print ("   Columns: %s (%s num, %s cat, %s date)" % (df.shape[1], 
                                                    len(numerical_columns), len(categorical_columns),
                                                    len(date_columns)) )


# In[8]:

partition20num=partition20[numerical_columns]


# In[9]:

partition20num.shape


# In[13]:

partition20num.columns


# In[14]:

partition20num.iloc[1:10,]


# In[15]:

part20mapdict=dict()
part20mapdict['id']=partition20num.id
part20mapdict['lat']=partition20num.latitude
part20mapdict['lon']=partition20num.longitude


# In[17]:

mapdf=pd.DataFrame(part20mapdict)


# In[19]:

mapdf.iloc[1:10,:]


# In[28]:

import geoplotlib
from geoplotlib.colors import colorbrewer
from geoplotlib.utils import epoch_to_str, BoundingBox, read_csv


# In[32]:

mapdf.to_csv('/Users/huyuxuan/desktop/finalproject/map.csv',index=False)


# In[55]:

mapdf1=mapdf.iloc[1:50,:]
mapdf2=mapdf.iloc[400:450,:]


# In[60]:

get_ipython().magic('pinfo geoplotlib')


# In[61]:

geoplotlib.dot(mapdf)
geoplotlib.dot(mapdf2)
geoplotlib.show()


# In[66]:

geoplotlib.dot(mapdf)
geoplotlib.kde(mapdf, bw=5)
geoplotlib.set_bbox(BoundingBox.KBH)
geoplotlib.show()


# In[57]:

geoplotlib.dot(mapdf1)
geoplotlib.hist(mapdf1,colorscale='sqrt',binsize=8)
geoplotlib.hist(mapdf2,colorscale='sqrt',binsize=2)
geoplotlib.show()


# In[40]:

geoplotlib.kde(mapdf,bw=[5,5])
geoplotlib.show()


# In[42]:

thedata = geoplotlib.utils.read_csv('/Users/huyuxuan/desktop/finalproject/map.csv')
geoplotlib.dot(thedata)
geoplotlib.inline()


# In[43]:

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
m = Basemap(width=12000000,height=9000000,projection='lcc',
            resolution=None,lat_1=45.,lat_2=55,lat_0=50,lon_0=-107.)
m.shadedrelief()
plt.show()


# In[44]:

import matplotlib.pyplot as plt
import matplotlib.cm
 
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize


# In[48]:

for feature in partition20num.columns:
    v = partition20num[feature].mean()
    if np.isnan(v):
        v = 0
    print ("Filling %s with %s" % (feature, v))
    partition20num[feature] = partition20num[feature].fillna(v)


# In[51]:

sum(partition20num.isnull().any())


# In[52]:

partition20num.corr()


# In[53]:

import seaborn as sns
fig = plt.figure(figsize=(50, 50))
corr = partition20num.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)


# In[54]:

partition20.shape


# In[ ]:

AIzaSyCi5xdffWfkap-iLP21FkGatZlSCZGrqgg


# In[101]:

import gmaps
import gmaps.datasets
gmaps.configure(api_key="AIzaSyCi5xdffWfkap-iLP21FkGatZlSCZGrqgg")


# In[74]:

mapdf.columns


# In[77]:

np.array(mapdf.lat,mapdf.lon).shape


# In[87]:

len(mapdf.lat)


# In[94]:

lonlatlist=[(mapdf.lat[i], mapdf.lon[i]) for i in range(291199)]


# In[ ]:

myarray=np.array([])


# In[97]:

myarray = np.array(lonlatlist)


# In[98]:

locations = gmaps.datasets.load_dataset("myarray")


# In[105]:

import folium
m = folium.Map(location=[(45.5236, -122.6750),(45.5237, -122.899)])
m


# In[107]:

import numpy as np
import matplotlib.pyplot as plt

plt.axes([0,0,1,1])

N = 20
theta = np.arange(0.0, 2*np.pi, 2*np.pi/N)
radii = 10*np.random.rand(N)
width = np.pi/4*np.random.rand(N)
bars = plt.bar(theta, radii, width=width, bottom=0.0)

for r,bar in zip(radii, bars):
    bar.set_facecolor(cm.jet(r/10.))
    bar.set_alpha(0.5)

plt.show()


# In[114]:

len(width)


# In[120]:

width


# In[118]:

import numpy as np
import matplotlib.pyplot as plt


# Fixing random state for reproducibility
np.random.seed(19680801)

# Compute pie slices
N = 20
theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
radii = 10 * np.random.rand(N)
width = np.pi / 4 * np.random.rand(N)

ax = plt.subplot(111, projection='polar')
bars = ax.bar(theta, radii, width=width, bottom=0.0)

# Use custom colors and opacity
for r, bar in zip(radii, bars):
    bar.set_facecolor(plt.cm.viridis(r / 10.))
    bar.set_alpha(0.5)

plt.show()


# In[119]:

theta


# In[ ]:



