
import pandas as pd
import numpy as np
from __future__ import print_function
from sklearn.cluster import KMeans

def plot_inertia(km, X, n_cluster_range):
    inertias = []
    for i in n_cluster_range:
        km.set_params(n_clusters=i)
        km.fit(X)
        inertias.append(km.inertia_)
    plt.plot(n_cluster_range, inertias, marker='o')
    plt.title('Elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()

vertical=pd.read_csv('/Users/huyuxuan/desktop/vertical.csv',header=None)
vertical[vertical.subcategory=='Age']
partition20.F3753
partition20['label']=tkmrt20drp.labels_
partition20
partition20=pd.read_csv('/Users/huyuxuan/desktop/20.csv')
partition20.shape
predcols=pd.read_csv('/Users/huyuxuan/desktop/predictcol1.csv',header=None)
predcolist=predcols[0].tolist()
partition20.shape
vertical.columns=["colName", "category", "subcategory"]

retailcols=vertical.loc[vertical['category']=='Retail','colName']

type(retailcols)

retailcolslist=retailcols.tolist()
print(retailcolslist)
retaildf20=partition20[retailcolslist]


# In[36]:

retaildf20.shape


# In[16]:

retaildf20.head()


# In[37]:

sum(retaildf20.isnull().any())


# In[38]:

len((retaildf20 == 0).any().tolist())


# In[15]:

retaildf20col0=(retaildf20 == 0).any().tolist()


# In[16]:

get_ipython().magic('pylab inline')
import pandas as pd, numpy as np             # Data manipulation 
from sklearn.decomposition import PCA        # The main algorithm
from matplotlib import pyplot as plt         # Graphing
import seaborn as sns                        # Graphing
from collections import defaultdict, Counter # Utils
sns.set(style="white")                       # Tuning the style of charts
import warnings                              # Disable some warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)


# In[33]:

dataset_limit = 300000
keep_dates = False


# In[17]:

# Take a handle on the dataset
mydataset = retaildf20

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


# In[25]:

#'F333' in numerical_columns


# In[18]:

newlist=list()
for i in predcolist:
    if i in numerical_columns:
        newlist.append(i)


# In[19]:

print(newlist)


# In[20]:

category1=newlist[1:25]#category 1


# In[31]:

retaildf20test=retaildf20.copy()


# In[41]:

retaildf20test[category2]=100-retaildf20test[category2]


# In[43]:

retaildf20test[category2].head()


# In[22]:

category2=newlist[25:38]#category 2 1-likely


# In[23]:

category3=newlist[39:77] #category3 1-likely


# In[44]:

retaildf20test[category3].head()


# In[42]:

retaildf20test[category3]=10-retaildf20test[category3]


# In[45]:

retaildf20test[category1].head()


# In[36]:

retaildf20test=retaildf20test[numerical_columns]


# In[35]:

retaildf20test.shape


# In[33]:

retaildf20.shape


# In[32]:

#test=retaildf20[category1]


# In[37]:

print(category3)


# In[38]:

retaildf20test[category1]=retaildf20test[category1].replace(0,50)


# In[39]:

retaildf20test[category2]=retaildf20test[category2].replace(0,50)


# In[40]:

retaildf20test[category3]=retaildf20test[category3].replace(0,5)


# In[34]:

retaildf20test.shape


# In[46]:

# Use mean for numerical features
for feature in retaildf20test.columns:
    v = retaildf20test[feature].mean()
    if np.isnan(v):
        v = 0
    print ("Filling %s with %s" % (feature, v))
    retaildf20test[feature] = retaildf20test[feature].fillna(v)
    


# In[55]:

retaildf20test.shape


# In[54]:

sum(retaildf20test.isnull().any())


# In[55]:

retaildf20test.to_csv('/Users/huyuxuan/desktop/finalproject/rtafterimpute.csv',index=False)


# In[2]:

retaildf20test=pd.read_csv('/Users/huyuxuan/desktop/finalproject/rtafterimpute.csv',header=None)


# In[45]:

retaildf20test.shape


# In[ ]:

#######finish the first part##########next is draft#################


# In[25]:

# Use mode for categorical features
for feature in categorical_columns:
    v = df[feature].value_counts().index[0]
    df[feature] = df[feature].fillna(v)


# In[29]:

X = dfnumeric.values
from sklearn.preprocessing import StandardScaler
ss = StandardScaler().fit(X)
X_std = ss.transform(X)


# In[30]:

pca = PCA()


# In[31]:

type(X)


# In[32]:

pca.set_params(n_components=None)
pca.fit(X_std)
plt.plot(range(len(dfnumeric.columns)), pca.explained_variance_ratio_)
plt.scatter(range(len(dfnumeric.columns)), pca.explained_variance_ratio_)
plt.xlabel('ith components')
plt.ylabel('Percentage of Variance')
plt.show()


# In[4]:

#90% of variance
pca.fit(X_std)
i=0; sumVar=0
while sumVar < .80:
    sumVar += pca.explained_variance_ratio_[i]
    print('component{0}, Variance explained = {1}'.format(i, round(sumVar, 3)))
    i+=1


# In[5]:

#80% of variance importance 
pca.fit(X_std)
i=0; sumVar=0
while sumVar < .80:
    sumVar += pca.explained_variance_ratio_[i]
    print('component{0}, Variance explained = {1}'.format(i, round(sumVar, 3)))
    i+=1


# In[36]:

sklearn_pca = PCA()
Y_sklearn = sklearn_pca.fit_transform(X_std)


# In[37]:

VARIANCE_TO_KEEP = 0.9


# In[38]:

plt.bar(range(sklearn_pca.n_components_), sklearn_pca.explained_variance_ratio_, alpha=0.5, align='center',label='individual explained variance')
plt.step(range(sklearn_pca.n_components_), [sklearn_pca.explained_variance_ratio_[:y].sum() for y in range(1,sklearn_pca.n_components_+1)], alpha=0.5, where='mid',label='cumulative explained variance')
plt.axhline(y=0.95, linewidth=2, color = 'r')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.xlim([0, sklearn_pca.n_components_])
plt.legend(loc='best')
plt.tight_layout()

keep_recommend = [sklearn_pca.explained_variance_ratio_[:y].sum()>VARIANCE_TO_KEEP for y in range(1,sklearn_pca.n_components_+1)].count(False)
print ("Number of components to keep to retain %s%% of the variance:" % (100*VARIANCE_TO_KEEP), keep_recommend, "out of the original", sklearn_pca.n_components_)






# In[39]:

retained_components_number = keep_recommend


# In[40]:

sklearn_pca_final = PCA(n_components=retained_components_number)
Y_sklearn_final = sklearn_pca_final.fit_transform(X_std)


# In[121]:

pcafit=PCA(n_components=77)
finaly=pcafit.fit_transform(rt20_std)


# In[128]:

pcafit.n_components_


# In[130]:

rt20_std.shape


# In[131]:

n_components_to_show = min(77, pcafit.n_components_)
n_input_features = pcafit.components_.shape[1]

decomp_df = pd.DataFrame(pcafit.components_[0:n_components_to_show],
                            columns=rt20droptest.columns[0:230])
if decomp_df.shape[1] > 77:
    decomp_df = decomp_df[decomp_df.columns[0:77]]

fig = plt.figure(figsize=(n_input_features, n_components_to_show))
sns.set(font_scale=3)
sns.heatmap(decomp_df, square=True)
sns.set(font_scale=1)


# In[ ]:




# In[ ]:




# In[65]:

n_components_to_show


# In[73]:

len(sklearn_pca_final.components_)


# In[41]:

##for the final model: 132 components
sklearn_pca_final.n_components_


# In[51]:

max(sklearn_pca_final.components_[1])


# In[52]:

min(sklearn_pca_final.components_[1])


# In[97]:

sklearn_pca_final.components_


# In[107]:

len(range(132))


# In[108]:

plot(range(132),sklearn_pca_final.components_[:,1])


# In[42]:

sklearn_pca_final.explained_variance_ratio_


# In[60]:

plt.scatter(range(1, 133), np.cumsum(sklearn_pca_final.explained_variance_ratio_))
plt.plot(range(1, 133), np.cumsum(sklearn_pca_final.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('Cumulative percentage of variance')
plt.xlim(0, 140)
plt.show()


# In[113]:

# For display reasons, we don't show all components if more than 50 (same for input variables)
#choose 132 components for pca final model
n_components_to_show = min(50, sklearn_pca_final.n_components_)
#n_components_to_show = sklearn_pca_final.n_components_
n_input_features = sklearn_pca_final.components_.shape[1]

decomp_df = pd.DataFrame(sklearn_pca_final.components_[0:n_components_to_show],
                            columns=dfnumeric.columns)
#if decomp_df.shape[1] > 50:
#    decomp_df = decomp_df[decomp_df.columns[0:50]]

fig = plt.figure(figsize=(n_input_features, n_components_to_show))
sns.set(font_scale=3)
sns.heatmap(decomp_df, square=True)
sns.set(font_scale=1)


# In[110]:

# For display reasons, we don't show all components if more than 50 (same for input variables)
#choose 132 components for pca final model
n_components_to_show = min(50, sklearn_pca_final.n_components_)
#n_components_to_show = sklearn_pca_final.n_components_
n_input_features = sklearn_pca_final.components_.shape[1]

decomp_df = pd.DataFrame(sklearn_pca_final.components_[0:n_components_to_show],
                            columns=dfnumeric.columns)
if decomp_df.shape[1] > 50:
    decomp_df = decomp_df[decomp_df.columns[0:50]]

fig = plt.figure(figsize=(n_input_features, n_components_to_show))
sns.set(font_scale=3)
sns.heatmap(decomp_df, square=True)
sns.set(font_scale=1)


# In[112]:

# For display reasons, we don't show all components if more than 50 (same for input variables)
#choose 132 components for pca final model
n_components_to_show = min(50, sklearn_pca_final.n_components_)
#n_components_to_show = sklearn_pca_final.n_components_
n_input_features = sklearn_pca_final.components_.shape[1]

decomp_df = pd.DataFrame(sklearn_pca_final.components_[n_components_to_show:n_components_to_show+50],
                            columns=dfnumeric.columns)
if decomp_df.shape[1] > 50:
    decomp_df = decomp_df[decomp_df.columns[50:50+50]]

fig = plt.figure(figsize=(n_input_features, n_components_to_show))
sns.set(font_scale=3)
sns.heatmap(decomp_df, square=True)
sns.set(font_scale=1)


# In[46]:

if len(numerical_columns) >= 2:
    feat1 = numerical_columns[0]
    feat2 = numerical_columns[1]
else:
    raise ValueError("Failed to automatically select proper variables to plot, please select manually")
    
print ("Will plot on these two features: '%s' and '%s'" % (feat1, feat2))


# In[47]:

idx_feat_1 = list(dfnumeric.columns).index(feat1)
idx_feat_2 = list(dfnumericsamp=1000


# In[53]:

list(zip(sklearn_pca_final.explained_variance_ratio_, sklearn_pca_final.components_))[0:2]


# In[115]:

####use the pca to a new dataset
df_PCA = pd.DataFrame(Y_sklearn_final, columns=[("PCA_component_" + str(comp)) for comp in range(sklearn_pca_final.n_components)])


# In[116]:

df_PCA.head()


# In[117]:

###cluster 
from __future__ import print_function
from sklearn.cluster import KMeans
kmeans = KMeans()


# In[119]:

Y_sklearn_final.shape


# In[120]:

kmeans.set_params(n_clusters=2)
kmeans.fit(Y_sklearn_final)


# In[121]:

kmeans.cluster_centers_


# In[122]:

kmeans.labels_


# In[123]:

def plot_inertia(km, X, n_cluster_range):
    inertias = []
    for i in n_cluster_range:
        km.set_params(n_clusters=i)
        km.fit(X)
        inertias.append(km.inertia_)
    plt.plot(n_cluster_range, inertias, marker='o')
    plt.title('Elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()


# In[124]:

plot_inertia(kmeans, Y_sklearn_final, range(1, 10))


# In[ ]:

plot_inertia(kmeans, Y_sklearn_final, range(1, 100))


# In[65]:

################# keep working on the next part ################


# In[56]:

X = retaildf20test.values
from sklearn.preprocessing import StandardScaler
ss = StandardScaler().fit(X)
X_std = ss.transform(X)


# In[57]:

pcaretail = PCA()


# In[58]:

pcaretail.set_params(n_components=None)
pcaretail.fit(X_std)
plt.plot(range(len(retaildf20test.columns)), pcaretail.explained_variance_ratio_)
plt.scatter(range(len(retaildf20test.columns)), pcaretail.explained_variance_ratio_)
plt.xlabel('ith components')
plt.ylabel('Percentage of Variance')
plt.show()


# In[68]:

pcaretail.fit(retaildf20test)
i=0; sumVar=0
while sumVar < .80:
    sumVar += pcaretail.explained_variance_ratio_[i]
    print('component{0}, Variance explained = {1}'.format(i, round(sumVar, 3)))
    i+=1


# In[59]:

pcaretail.fit(retaildf20test)
i=0; sumVar=0
while sumVar < .90:
    sumVar += pcaretail.explained_variance_ratio_[i]
    print('component{0}, Variance explained = {1}'.format(i, round(sumVar, 3)))
    i+=1


# In[61]:

retaildf20_pca_final = PCA(n_components=21)
retaildf20_final = retaildf20_pca_final.fit_transform(X_std)


# In[77]:

retaildf15_pca_final = PCA(n_components=15)
retaildf15_final = retaildf15_pca_final.fit_transform(X_std)


# In[112]:

retaildf10_pca_final = PCA(n_components=10)
retaildf10_final = retaildf10_pca_final.fit_transform(X_std)


# In[113]:

retaildf10_pca_final.n_components


# In[78]:

retaildf15_pca_final.n_components


# In[79]:

retaildf15_final.shape


# In[114]:

retaildf10pca = pd.DataFrame(retaildf10_final, columns=[("PCA_component_" + str(comp)) for comp in range(retaildf10_pca_final.n_components)])


# In[80]:

retaildf15pca = pd.DataFrame(retaildf15_final, columns=[("PCA_component_" + str(comp)) for comp in range(retaildf15_pca_final.n_components)])


# In[81]:

retaildf15pca.to_csv('/Users/huyuxuan/desktop/finalproject/retaildf15.csv',index=False)


# In[62]:

retaildf20_pca_final.n_components


# In[63]:

retaildf20pca = pd.DataFrame(retaildf20_final, columns=[("PCA_component_" + str(comp)) for comp in range(retaildf20_pca_final.n_components)])


# In[64]:

retaildf20pca.to_csv('/Users/huyuxuan/desktop/finalproject/retaildf20.csv',index=False)


# In[65]:

retaildf20pca.shape


# In[66]:

retaildf20pca.iloc[1:10,]


# In[67]:

retaildf20pca.std(axis=0)


# In[ ]:

############### start clustering ##################


# In[44]:

#use new dataset to do the clustering
from __future__ import print_function
from sklearn.cluster import KMeans
kmeanspart20 = KMeans()


# In[45]:

kmeanspart20.set_params(n_clusters=2)
kmeanspart20.fit(retaildf20pca)


# In[77]:

from sklearn.cluster import MiniBatchKMeans,KMeans


# In[152]:

def plot_inertia(km, X, n_cluster_range):
    inertias = []
    for i in n_cluster_range:
        km.set_params(n_clusters=i)
        km.fit(X)
        inertias.append(km.inertia_)
    plt.plot(n_cluster_range, inertias, marker='o')
    plt.title('Elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()


# In[155]:

plot_inertia(kmeanspart20, retaildf20pca, range(1, 20))


# In[ ]:

kmeanspart20newK = MiniBatchKMeans()


# In[ ]:

kmeanspart20 = KMeans(n_jobs=-1)


# In[75]:

kmeanspart20k10


# In[47]:

#choose the number of clustering
kmeanspart20k10 = KMeans()
kmeanspart20k10.set_params(n_clusters=10)
kmeanspart20k10.fit(retaildf20pca)


# In[76]:

kmeanspart20k10.cluster_centers_


# In[77]:

len(kmeanspart20k10.labels_)


# In[49]:

retaildf20pca['clusterfeature']=kmeanspart20k10.labels_


# In[162]:

retaildf20pca.shape


# In[50]:

rtclstrk10c=pd.DataFrame(kmeanspart20k10.cluster_centers_)


# In[51]:

rtclstrk10c.shape


# In[52]:

rtclstrk10c


# In[173]:

len(kmeanspart20k10.cluster_centers_)


# In[83]:

plt.scatter(retaildf20pca.iloc[:,1], retaildf20pca.iloc[:,15], c=kmeanspart20k10.labels_, alpha=0.8,cmap='plasma')


# In[82]:

plt.scatter(retaildf20pca.iloc[:,10], retaildf20pca.iloc[:,11], c=kmeanspart20k10.labels_, alpha=0.8,cmap='plasma')


# In[81]:

plt.scatter(retaildf20pca.iloc[:,7], retaildf20pca.iloc[:,8], c=kmeanspart20k10.labels_, alpha=0.8,cmap='plasma')


# In[80]:

plt.scatter(retaildf20pca.iloc[:,6], retaildf20pca.iloc[:,7], c=kmeanspart20k10.labels_, alpha=0.8,cmap='plasma')


# In[79]:

plt.scatter(retaildf20pca.iloc[:,5], retaildf20pca.iloc[:,6], c=kmeanspart20k10.labels_, alpha=0.8,cmap='plasma')


# In[175]:

plt.scatter(retaildf20pca.iloc[:,0], retaildf20pca.iloc[:,1], c=kmeanspart20k10.labels_, alpha=0.8,cmap='plasma')


# In[176]:

plt.scatter(retaildf20pca.iloc[:,2], retaildf20pca.iloc[:,3], c=kmeanspart20k10.labels_, alpha=0.8,cmap='plasma')


# In[177]:

plt.scatter(retaildf20pca.iloc[:,3], retaildf20pca.iloc[:,4], c=kmeanspart20k10.labels_, alpha=0.8,cmap='plasma')


# In[179]:

plt.scatter(retaildf20pca.iloc[:,4], retaildf20pca.iloc[:,5], c=kmeanspart20k10.labels_, alpha=0.8,cmap='plasma')


# In[1]:

from matplotlib import cm
from sklearn.metrics import silhouette_samples

x=retaildf20pca
y_km = kmeanspart20k10.fit_predict(x)
cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(x, y_km, metric='euclidean')
y_ax_lower, y_ax_upper = 0, 0
yticks = []
for i, c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[y_km == c]
    c_silhouette_vals.sort()
    size_cluster_c = len(c_silhouette_vals)
    y_ax_upper += size_cluster_c
    color = cm.jet(1.0* i / n_clusters)
    plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0, color=color)
    yticks.append((y_ax_lower + y_ax_upper) / 2)
    y_ax_lower += size_cluster_c
    silhouette_avg = np.mean(silhouette_vals)
    plt.axvline(silhouette_avg, color='red', linestyle='--')
    plt.yticks(yticks, cluster_labels + 1)
    plt.title('Silhouette Analysis')
    plt.ylabel('Cluster')
    plt.xlabel('Silhouette coefficient')
    plt.show()


# In[85]:

retaildf20test.shape


# In[ ]:

############### try for clustering using minibatch


# In[38]:

##second try for cluster 
from sklearn.cluster import MiniBatchKMeans,KMeans
kmeanretail= MiniBatchKMeans()
kmeanretail.set_params(n_clusters=10)
kmeanretail.fit(retaildf20test)


# In[39]:

kmeanretailn7= MiniBatchKMeans()
kmeanretailn7.set_params(n_clusters=7)
kmeanretailn7.fit(retaildf20test)


# In[145]:

clustern7centerdf=retaildf10_pca_final.transform(kmeanretailn7.cluster_centers_)


# In[83]:

kmeanretail.cluster_centers_


# In[84]:

kmeanretail.labels_


# In[115]:

retaildf10pca.head()


# In[125]:

retaildf10pca['clustern7']=kmeanretailn7.labels_


# In[127]:

x=retaildf10pca.loc[retaildf10pca.PCA_component_0<20,:]
x=x.loc[x.PCA_component_1<40]


# In[157]:

max(x.PCA_component_1)


# In[128]:

x.head()


# In[131]:

plt.scatter(retaildf10pca.iloc[:,0], retaildf10pca.iloc[:,1], c=kmeanretailn7.labels_, alpha=0.8,cmap='plasma')


# In[148]:




# In[149]:

plt.scatter(clustern7centerdf[0],clustern7centerdf[1],c='k',s=30)


# In[147]:

plt.scatter(x.iloc[:,0], x.iloc[:,1], c=x.clustern7, alpha=0.1,cmap='plasma',s=5)
plt.scatter(clustern7centerdf[0],clustern7centerdf[1],c='k',s=30)
plt.show()


# In[ ]:

from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import random


fig = pyplot.figure()
ax = Axes3D(fig)

x = retaildf20pca.iloc[:,0]
y = retaildf20pca.iloc[:,1]
z = retaildf20pca.iloc[:,2]

random.shuffle(x)
random.shuffle(y)
random.shuffle(z)

ax.scatter(x,y,z,c=kmeanretail.labels_,alpha=0.8,cmap='plasma')
pyplot.show()


# In[ ]:

type(retaildf20pca.iloc[:,0])


# In[ ]:

retaildf20test.to_csv('/Users/huyuxuan/desktop/finalproject/retaildf20original.csv',index=False)


# In[160]:

retaildf10pca1=retaildf10pca.iloc[:,1:10]


# In[161]:

retaildf10pca1.head()


# In[162]:

X = retaildf10pca1.values
from sklearn.preprocessing import StandardScaler
ss = StandardScaler().fit(retaildf10pca1)
X_stdpca10 = ss.transform(X)


# In[164]:

X_stdpca10


# In[167]:

scaled_PCA = pd.DataFrame(X_stdpca10, columns=[("PCA_component_" + str(comp)) for comp in range(9)])


# In[168]:

scaled_PCA['clustern7']=kmeanretailn7.labels_


# In[169]:

plt.scatter(scaled_PCA.iloc[:,0], scaled_PCA.iloc[:,1], c=scaled_PCA.clustern7, alpha=0.1,cmap='plasma',s=5)


# In[172]:

retaildf20test.shape


# In[175]:

retaildf20test.describe()


# In[176]:

type(retaildf20test.describe())


# In[180]:

retaildf20test.describe().iloc[:,0:20]


# In[186]:

retaildf20test.describe().iloc[:,20:40]


# In[187]:

retaildf20test.describe().iloc[:,40:60]


# In[188]:

retaildf20test.describe().iloc[:,60:80]


# In[189]:

retaildf20test.describe().iloc[:,80:100]


# In[190]:

retaildf20test.describe().iloc[:,100:120]


# In[191]:

retaildf20test.describe().iloc[:,120:140]


# In[192]:

retaildf20test.describe().iloc[:,140:160]


# In[193]:

retaildf20test.describe().iloc[:,160:180]


# In[47]:

rt20droptest=retaildf20test.copy()


# In[59]:

rt20droptest.shape


# In[50]:

print(datecolumns)


# In[49]:

rt20droptest=rt20droptest.drop(rt20droptest[datecolumns],axis=1)


# In[50]:

rt20droptest.shape


# In[194]:

retaildf20test.describe().iloc[:,180:200]


# In[195]:

retaildf20test.describe().iloc[:,200:220]


# In[48]:

datecolumns=['F3925','F3929','F3933','F3937','F3953','F3957','F3961','F3965','F3969','F3973','F3977',             'F3981','F3985','F3989','F3993','F3997','F4001','F4005','F5009','F5013','F7017','F7021',             'F7025','F7029','F7033','F7037','F7041','F7045','F7049','F7053','F7057','F7061','F7065',             'F7069','F7073','F7077','F7081','F7085']


# In[57]:

retaildf20test


# In[196]:

retaildf20test.describe().iloc[:,220:240]


# In[197]:

retaildf20test.describe().iloc[:,240:260]


# In[198]:

retaildf20test.describe().iloc[:,260:]


# In[208]:

len(category3)


# In[51]:

# category 3 ten times larger than before
rt20droptest[category3]=rt20droptest[category3]*10


# In[211]:

vertical


# In[222]:

musiclist=vertical[vertical.category=='Retail'].loc[vertical.subcategory=='Misc','colName'].tolist()


# In[250]:

mslist=[]
for ms in musiclist:
    if ms in rt20droptest:
        mslist.append(ms)
print(mslist)


# In[251]:

rt20droptest[mslist].describe().iloc[:,1:15]


# In[252]:

rt20droptest[mslist].describe().iloc[:,15:]


# In[309]:

msdf=rt20droptest[mslist]


# In[310]:

msdf.shape


# In[311]:

print(msdf['F7015'].tolist())


# In[280]:

sum(rt20droptest==9997)


# In[291]:

sum(msdf==9999)


# In[258]:

msdf.columns


# In[317]:

partition20.loc[219838,'F14695']


# In[314]:

print(category2)


# In[312]:

msdf.loc[msdf.F7015==9999,:].iloc[:,1:20]


# In[287]:

msdftest=msdf.replace(9999,0)


# In[289]:

msdftest.describe().iloc[:,1:15]


# In[290]:

msdftest.describe().iloc[:,15:]


# In[52]:

rt20droptest.shape


# In[126]:

rt20droptest.iloc[1:10,:]


# In[124]:

rt20droptest.shape


# In[123]:

pcart20drp


# In[53]:

rt20sd = rt20droptest.values
from sklearn.preprocessing import StandardScaler
rt20sdss = StandardScaler().fit(rt20sd)
rt20_std = rt20sdss.transform(rt20sd)
rt20_std


# In[48]:

rt20droptest.shape


# In[71]:

pcart20drp = PCA()
pcart20drp.set_params(n_components=None)
pcart20drp.fit(rt20_std)
plt.plot(range(len(rt20droptest.columns)), pcart20drp.explained_variance_ratio_)
plt.scatter(range(len(rt20droptest.columns)), pcart20drp.explained_variance_ratio_)
plt.xlabel('ith components')
plt.ylabel('Percentage of Variance')
plt.show()


# In[118]:

pcart20drp


# In[73]:

pcart20drp.fit(rt20_std)
i=0; sumVar=0
while sumVar < .80:
    sumVar += pcart20drp.explained_variance_ratio_[i]
    print('component{0}, Variance explained = {1}'.format(i, round(sumVar, 3)))
    i+=1


# In[64]:

rt20_std.shape


# In[65]:

kmrt20drp.set_params(n_clusters=20)
kmrt20drp.fit(rt20_std)


# In[49]:

kmrt20drp= MiniBatchKMeans()


# In[115]:

def plot_inertia(km, X, n_cluster_range):
    inertias = []
    for i in n_cluster_range:
        km.set_params(n_clusters=i)
        km.fit(X)
        inertias.append(km.inertia_)
    plt.plot(n_cluster_range, inertias, marker='o')
    plt.title('Elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()
    


# In[52]:

plot_inertia(kmrt20drp,rt20_std,range(1,40))


# In[65]:

plot_inertia(kmrt20drp,rt20_std,range(1,40))


# In[92]:

tkmrt20drp = KMeans()


# In[339]:

plot_inertia(tkmrt20drp,rt20_std,range(1,10))


# In[93]:

tkmrt20drp.set_params(n_clusters=7)
tkmrt20drp.fit(rt20_std)


# In[94]:

tkmrt20drp


# In[342]:

rtdroppca75 = PCA(n_components=75)
rtdroppcafinal = rtdroppca75.fit_transform(rt20_std)


# In[81]:

rtdroppca75 = PCA(n_components=3)
rtdroppcafinal = rtdroppca75.fit_transform(rt20_std)


# In[82]:

rt77df = pd.DataFrame(rtdroppcafinal, columns=[("PCA_component_" + str(comp)) for comp in range(rtdroppca75.n_components)])


# In[83]:

rt77df.shape


# In[345]:

plt.scatter(rt77df.iloc[:,0],rt77df.iloc[:,1], c=tkmrt20drp.labels_, alpha=0.8,cmap='plasma')


# In[95]:

plt.scatter(rt77df.iloc[:,0],rt77df.iloc[:,1], c=tkmrt20drp.labels_, alpha=0.8,cmap='plasma')


# In[89]:

len(kmrt20drp.labels_)


# In[371]:

rt77df.shape


# In[97]:

plt.scatter(rt77df.iloc[0:90000,0],rt77df.iloc[0:90000,1], c=kmrt20drp.labels_[0:90000], alpha=0.8,cmap='plasma')


# In[99]:

plt.scatter(rt77df.iloc[0:9000,0],rt77df.iloc[0:9000,1], c=tkmrt20drp.labels_[0:9000], alpha=0.8,cmap='plasma')


# In[100]:

plt.scatter(rt77df.iloc[0:5000,0],rt77df.iloc[0:5000,1], c=tkmrt20drp.labels_[0:5000], alpha=0.8,cmap='plasma')


# In[110]:

plt.scatter(rt77df.iloc[0:3000,0],rt77df.iloc[0:3000,1], c=tkmrt20drp.labels_[0:3000], alpha=0.8,cmap='plasma')


# In[ ]:




# In[ ]:




# In[113]:

from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import random


fig = pyplot.figure()
ax = Axes3D(fig)

x = rt77df.iloc[0:10000,0]
y = rt77df.iloc[0:10000,1]
z = rt77df.iloc[0:10000,2]

random.shuffle(x)
random.shuffle(y)
random.shuffle(z)

ax.scatter(x,y,z,c=tkmrt20drp.labels_[0:10000],alpha=0.5,cmap='plasma')
pyplot.show()


# In[96]:

rt20droptest.shape


# In[ ]:

#tkmrt20drp = KMeans(n_jobs=-1)
#plot_inertia(tkmrt20drp,rt20_std,range(1,15))


# In[54]:

tkmrt20drp = KMeans()
tkmrt20drp.set_params(n_clusters=7)
tkmrt20drp.fit(rt20_std)


# In[55]:

cluster7label=tkmrt20drp.labels_


# In[279]:

type(cluster7label)


# In[77]:

vertical.columns


# In[78]:

vertical


# In[80]:

vertical.loc[vertical.category=='Retail',:].groupby('subcategory').count()


# In[56]:

rt20droptest['label']=cluster7label


# In[81]:

rt20droptest.head(10)


# In[59]:

rt20droptest.iloc[1:10,]


# In[57]:

import numpy as np
import pandas as pd
import missingno as msno
import re
get_ipython().magic('matplotlib inline')


# In[61]:

rt20droptest.corr()


# In[63]:

import seaborn as sns
fig = plt.figure(figsize=(50, 50))
corr = rt20droptest.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)


# In[64]:

rt20droptest.iloc[1:10,]


# In[58]:

meank7df=rt20droptest.groupby('label').mean()


# In[84]:

meank7df.head(10)


# In[290]:

agey=partition20.groupby('label').mean().F3753


# In[304]:

agey/100


# In[297]:

len(partition20.groupby('label').mean().index)


# In[352]:

performance


# In[353]:

y_pos


# In[356]:

width


# In[360]:

y_pos/50


# In[365]:

sqrt(obser)/50


# In[ ]:

theta


# In[370]:

import numpy as np
import matplotlib.pyplot as plt


# Fixing random state for reproducibility
np.random.seed(19680801)

# Compute pie slices
#N = 7
theta = sqrt(obser)/500
radii = 10 * np.random.rand(N)
width = (np.pi/4)*(y_pos/50)

ax = plt.subplot(111, projection='polar')
bars = ax.bar(performance, y_pos, width=theta, bottom=0.0)

# Use custom colors and opacity
for r, bar in zip(radii, bars):
    bar.set_facecolor(plt.cm.viridis(r / 10.))
    bar.set_alpha(0.5)

plt.show()


# In[305]:

import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
#objects = ('Python', 'C++', 'Java', 'Perl', 'Scala', 'Lisp')
y_pos = agey/100
performance = partition20.groupby('label').mean().index
 
plt.bar(performance, y_pos, align='center', alpha=0.5,color=obser)
#plt.xticks(y_pos, objects)
#plt.ylabel('Usage')
#plt.title('Programming language usage')
 
plt.show()


# In[81]:

meank7df['label']=meank7df.index


# In[82]:

meank7df=meank7df.reset_index(drop=True)


# In[87]:

rt20droptest.shape


# In[116]:

meank7df.sort_values('F995')


# In[80]:

meank7df


# In[59]:

versemeank7df=meank7df.T


# In[60]:

versemeank7df.head(10)


# In[61]:

versemeank7df['colName']=versemeank7df.index


# In[62]:

versemeank7df


# In[63]:

versemeank7df=versemeank7df.reset_index(drop=True)


# In[64]:

meanall=pd.merge(versemeank7df,vertical,on='colName')


# In[ ]:




# In[114]:

meanalldrop.head(10)


# In[66]:

meanalldrop=meanalldrop.drop('colName', axis=1)


# In[65]:

meanalldrop=meanall.drop('category',axis=1)


# In[67]:

meanalldrop.columns


# In[68]:

meanalldrop['subcategory']=meanalldrop['subcategory'].replace('Pet supply','Pet Supply')


# In[93]:

meanalldrop['subcategory']=meanalldrop['subcategory'].replace("Travel (Airlines, Hotels, Domestic v. International)",                                                              'Travel')


# In[94]:

finaltable=meanalldrop.groupby('subcategory').mean()
finaltable



my_plot = finaltable.plot(kind='bar',stacked=True,title="Subcategory for seven groups of customers",figsize=(20, 17))
my_plot.set_xlabel("Customers")
my_plot.set_ylabel("Sales")
sns.set(font_scale=3)


from __future__ import print_function  # Python 2 and 3
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from scipy import stats
get_ipython().magic('matplotlib')
inlinemy_plot = category_group.unstack().plot(kind='bar',stacked=True,title="Total Sales by Customer")
my_plot.set_xlabel("Customers")
my_plot.set_ylabel("Sales")







plt.figure(figsize=(9, 6))
num_plots = 26
columns26=meank7df.columns[:26]
colormap = plt.cm.gist_ncar
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, num_plots)])
for i in columns26:
    plt.plot(meank7df.label, meank7df[i],alpha=0.8)
sns.set(font_scale=1)
plt.show()



len(meank7df.columns[26:65])



plt.figure(figsize=(9, 6))
num_plots = 39
columns39=meank7df.columns[26:65]
colormap = plt.cm.gist_ncar
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, num_plots)])
for i in columns39:
    plt.plot(meank7df.label, meank7df[i],alpha=0.8)

plt.show()



plt.figure(figsize=(9, 6))
num_plots = 14
columns39=meank7df.columns[26:40]
colormap = plt.cm.gist_ncar
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, num_plots)])
for i in columns39:
    plt.plot(meank7df.label, meank7df[i],alpha=0.8)
plt.show()

columnstype2=['F669','F685','F689','F690','F684','F681','F682','F687','F688','F790','F792','F791','F793',             'F707','F708','F703','F706','F705','F701','F702','F704']
columnstype3=['F700','F691','F694','F693','F686','F697','F695','F692','F683','F680','F699','F696','F698',             'F789','F709','F710','F794']

plt.figure(figsize=(9, 6))
columns=columnstype3
colormap = plt.cm.gist_ncar
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, len(columns))])
for i in columns:
    plt.plot(meank7df.label, meank7df[i],alpha=0.8)

plt.show()



plt.figure(figsize=(9, 6))
columns=columnstype2
colormap = plt.cm.gist_ncar
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, len(columns))])
for i in columns:
    plt.plot(meank7df.label, meank7df[i],alpha=0.8)
    
plt.show()



plt.figure(figsize=(9, 6))
num_plots = 2
columns=meank7df.columns[32:]
colormap = plt.cm.gist_ncar
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, len(columns))])
for i in columns39:
    plt.plot(meank7df.label, meank7df[i],alpha=0.8)
plt.show()



plt.figure(figsize=(9, 6))
num_plots = 5
columns39=meank7df.columns[26:31]
colormap = plt.cm.gist_ncar
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, num_plots)])
for i in columns39:
    plt.plot(meank7df.label, meank7df[i],alpha=0.8)
plt.show()


import folium
from branca.utilities import split_six

threshold_scale = split_six(partition20['label'])

m = folium.Map(location=[48, -102], zoom_start=3)

m.choropleth(
    geo_data=state_geo,
    data=partition20,
    columns=['State', 'Unemployment'],
    key_on='id',
    fill_color='BuPu',
    fill_opacity=0.7,
    line_opacity=0.5,
    legend_name='Unemployment Rate (%)',
    threshold_scale=threshold_scale,
    reset=True
)



import os
import folium
from folium.plugins.measure_control import MeasureControl

m = folium.Map(location=[38.850116, -77.511830], zoom_start=10)

c = MeasureControl()
c.add_to(m)

color=['blue','red','green','purple','yellow','grey','back']

for i in range(7):
    radius = 0.1*sd[i]
    folium.CircleMarker(
        location=[lat[i], long[i]],
        radius=radius,
        color='red',
        weight=2,
        fill_color=color[i],
        fill_opacity=0.6,
        opacity=1,
        fill=True,
        popup='Cluster {} group size {}'.format(i,obser[i]),
    ).add_to(m)
