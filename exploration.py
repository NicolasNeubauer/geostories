# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# Explore the generated dataset

# +
from collections import Counter


import pandas as pd, geopandas as gpd, json
from sklearn.cluster import KMeans
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import metrics 
from sklearn import preprocessing
from sklearn import tree
from sklearn import linear_model
from sklearn import feature_selection


import graphviz
# -

DATA_DIR = '/Users/nneubaue/Data/osm' 
INPUT_FILE = f"{DATA_DIR}/full_dataset_NYC.geojson"        # NYC taxi by taxi district
INPUT_FILE = f"{DATA_DIR}/full_dataset_california.geojson" # housing
INPUT_FILE = f"{DATA_DIR}/full_dataset_trips_NYC.geojson"  # NYC taxi by trip


def get_features(df, level, log, winsorized, normalized):
    def is_valid(column):
        if not column.startswith(f"level{level}"):
            return False
        if ("_w" in column) != winsorized:
            return False
        if ("_normalized" in column) != normalized:
            return False
        if ("_log" in column) != log:
            return False
        return True
    return [column for column in df.columns if is_valid(column)]


def transform_features(gdf, features=None, quantile_level=0.98, area_column="Shape_Area"):
    if features is None:
        features = [x for x in gdf.columns if x.startswith('level')]
    gdf_taxi_count_with_tag_count = gdf.copy() 
    for tag in features:
        if tag.startswith('level'):
            # windsorize
            quantile = gdf_taxi_count_with_tag_count[tag].fillna(0).quantile(quantile_level)
            gdf_taxi_count_with_tag_count[f"{tag}_w"] = \
                gdf_taxi_count_with_tag_count[tag].clip(upper=quantile)

            # normalize
            gdf_taxi_count_with_tag_count[f"{tag}_normalized"] = \
                gdf_taxi_count_with_tag_count[tag] / gdf_taxi_count_with_tag_count[area_column]

            # windsorize the normalized
            quantile = gdf_taxi_count_with_tag_count[f"{tag}_normalized"].fillna(0).quantile(quantile_level)
            gdf_taxi_count_with_tag_count[f"{tag}_normalized_w"] = \
                gdf_taxi_count_with_tag_count[f"{tag}_normalized"].clip(upper=quantile)

            # log the winsorized
            gdf_taxi_count_with_tag_count[f"{tag}_w_log"] = \
                np.log(gdf_taxi_count_with_tag_count[f"{tag}_w"]+1)

            # log the normalized value
            gdf_taxi_count_with_tag_count[f"{tag}_log_normalized"] = \
                np.log(gdf_taxi_count_with_tag_count[f"{tag}_normalized"]+1)

            # log the original value
            gdf_taxi_count_with_tag_count[f"{tag}_log"] = \
                np.log(gdf_taxi_count_with_tag_count[tag]+1)

            # log the normalized winsorized value
            gdf_taxi_count_with_tag_count[f"{tag}_log_normalized_w"] = \
                np.log(gdf_taxi_count_with_tag_count[f"{tag}_normalized_w"]+1)  
            
    return gdf_taxi_count_with_tag_count 


gdf = gpd.read_file(INPUT_FILE)
gdf = gdf.rename(columns = {col: col.replace('_', '-') if col.startswith('level') else col for col in gdf.columns})

list(gdf.columns)

# +
# 64000
gdf["center"] = \
   (gdf["num_dropoffs_20"]>34000) & \
   (gdf["num_dropoffs_20"]<189000) & \
   (gdf["do_relative"]<3.35)


gdf.plot(column="center")
# -

sns.relplot(data=gdf, x='level2-Sustenance', y='count_dropoff', hue='level2-Transportation', aspect=1.61)


sns.regplot(data=gdf, x='level2-Sustenance', y='count_dropoff', x_jitter=0.2, scatter_kws={'alpha': 0.02})


sns.scatterplot(data=df, x='count_pickup', y='count_dropoff', alpha=0.1)

gdf['count_pickup'].mean()

# +
#target = 'num_pickups_20'

df = gdf
#df = gdf[gdf['count_pickup']<8000]
#df = df[df['count_pickup']>100]
#df = df[df['num_dropoffs_20']>10000]

        
# transform features only after we've selected rows
df['ALAND'] = 10000 # resolution
df = transform_features(df, area_column="ALAND") 
#print(list(df.columns))

# -

target = 'pu_relative'
target = 'median_house_value'
target = 'count_pickup' # count_dropoff
target = 'count_dropoff'

# +

features = get_features(df, 2, log=True, winsorized=False, normalized=False)# + ['median_income']
#features = ['pca_1', 'pca_2']
#features = ['median_income']

X = np.array(df.fillna(0)[features])
poly = preprocessing.PolynomialFeatures(2)
X = poly.fit_transform(X)
features = poly.get_feature_names_out(features)



print('num features: ', len(features))

vt = feature_selection.VarianceThreshold().fit(X)
X = vt.transform(X)
feature_indices = [int(x[1:]) for x in vt.get_feature_names_out()]
features = [features[i] for i in feature_indices]
print(len(features), ' after 0 variance removed')

print('first 10 features:', list(features)[0:10], '...')

X = preprocessing.StandardScaler().fit_transform(X)
Y = df[target]

#X = feature_selection.SelectKBest(feature_selection.f_regression, k=20).fit_transform(X, Y)


print(f"MAE when predicting average: {(df[target]-df[target].mean()).abs().mean()}")
print()

reg = linear_model.LinearRegression().fit(X, Y)
print(f"r**2 of linear regression: {reg.score(X, Y)}")
print(f"MAE of linear regression: {metrics.mean_absolute_error(Y, reg.predict(X))}")

print()
MAX_DEPTH = 2
reg2 = tree.DecisionTreeRegressor(max_depth=MAX_DEPTH).fit(X, Y)
print(f"r**2 for tree of max depth {MAX_DEPTH}: {reg2.score(X, Y)}")
print(f"MAE for tree of max depth {MAX_DEPTH}: {metrics.mean_absolute_error(Y, reg2.predict(X))}")

from sklearn import linear_model
reg3 = linear_model.LassoLarsCV().fit(X, Y)
print()
print(f"r**2 of lasso regression: {reg3.score(X, Y)}")
print(f"MAE of lasso regression: {metrics.mean_absolute_error(Y, reg3.predict(X))}")

from sklearn import linear_model
reg4 = linear_model.Ridge(alpha=100).fit(X, Y)
print()
print(f"r**2 of ridge regression: {reg4.score(X, Y)}")
print(f"MAE of ridge regression: {metrics.mean_absolute_error(Y, reg4.predict(X))}")


# -

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X)

pca.transform(X).T

df_pca = pd.DataFrame(pca.transform(X), columns=['x1','x2'])
df_pca.plot.scatter(x='x1', y='x2', alpha=0.1)

df['pca_1'] = pca.transform(X).T[0]
df['pca_2'] = pca.transform(X).T[1]

sns.regplot(data=df, x='pca_1', y='count_pickup')

df['count_pickup_log'] = np.log(df['count_pickup'])

sns.scatterplot(data=df.sort_values('count_pickup_log', ascending=True), x='pca_1', y='pca_2', hue='count_pickup_log', alpha=0.3, palette='coolwarm')

pd.DataFrame(df.iloc[0][features]).reset_index().plot.bar(x='index')

df.sort_values('count_dropoff', ascending=False)[['index', 'count_dropoff'] + features ][-100:]

index=7728
row = df[df['index']==index].iloc[0][features]
pd.DataFrame(row).reset_index().plot.bar(x='index')

# +
features = get_features(df, 2, log=True, winsorized=False, normalized=False)# + ['median_income']
X2 = np.array(df.sort_values(target).fillna(0)[features])
vt = feature_selection.VarianceThreshold().fit(X2)
X2 = vt.transform(X2)
feature_indices = [int(x[1:]) for x in vt.get_feature_names_out()]
features = [features[i] for i in feature_indices]
print(len(features), ' after 0 variance removed')

X2 = preprocessing.StandardScaler().fit_transform(X2)
Y2 = df.sort_values(target)[target]
# -

plt.figure(figsize = (5,15))
plt.imshow(X2, cmap='hot', interpolation='nearest', aspect='auto')

features

X



sns.kdeplot(x=df['pca_1'], y=df['pca_2'])

sns.kdeplot(x=df['pca_2'], y=df['count_dropoff'])

print(pca.components_)
df_pca_features = pd.DataFrame(pca.components_.T)
df_pca_features
df_pca_features['name'] = features
df_pca_features.sort_values(1, ascending=False)

df

pd.DataFrame(zip(features, feature_selection.r_regression(X, Y)), columns=['feature', 'F-score']).sort_values('F-score', ascending=False)

df_coef = pd.DataFrame(zip(features, reg4.coef_), columns=['feature', 'coef'])
df_coef['coef_abs'] = df_coef['coef'].abs()
df_coef = df_coef[df_coef['coef_abs']>1]
print(len(df_coef))
df_coef.sort_values('coef_abs', ascending=False)

dot_data = tree.export_graphviz(reg2, out_file=None, 
                      feature_names=features,  
                      #class_names=iris.target_names,  
                      filled=True, rounded=True,  
                      special_characters=True)  
graph = graphviz.Source(dot_data)  
graph 

level2_features = [x for x in df.columns if x.startswith('level2-')]

gdf.plot(column="level2-Entertainment")

len(gdf)

df_features = pd.melt(df, id_vars=[], value_vars=level2_features).fillna(0)
df_features

df_features["p"] =\
  df_features.apply(lambda column: a[1] if len(a := column['variable'].split('_', 1)) > 1 else "", axis=1)

df_features["f"] = df_features.apply(lambda column: column['variable'].split('_')[0].split('-')[1], axis=1)

df_features['p'].unique()

g = sns.FacetGrid(df_features[~df_features['p'].str.contains('normalized')], 
                  row="f", 
                  col="p", 
                  sharex=False, 
                  sharey=False)
g.map(sns.histplot, "value")


