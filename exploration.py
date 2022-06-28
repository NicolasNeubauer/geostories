# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
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

from sklearn import metrics, preprocessing, tree, linear_model, feature_selection, model_selection, pipeline, dummy

import graphviz
# -

DATA_DIR = '/Users/nicolasneubauer/Data/geostories' 
INPUT_FILE = f"{DATA_DIR}/full_dataset_NYC.geojson"        # NYC taxi by taxi district
#INPUT_FILE = f"{DATA_DIR}/full_dataset_california.geojson" # housing
INPUT_FILE = f"{DATA_DIR}/full_dataset_trips_NYC_50.geojson"  # NYC taxi by trip
#INPUT_FILE = f"{DATA_DIR}/full_dataset_trips_NYC_CELLSIZE_100.geojson"  # NYC taxi by trip
LATLON_OUTPUT_FILE = f"{DATA_DIR}/full_dataset_trips_NYC_50_latlon.geojson"


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

sns.relplot(data=gdf, x='level2-Sustenance', y='count_dropoff', hue='level2-Transportation', aspect=1.61)


sns.regplot(data=gdf, x='level2-Sustenance', y='count_dropoff', x_jitter=0.2, scatter_kws={'alpha': 0.02})


sns.scatterplot(data=gdf, x='count_pickup', y='count_dropoff', alpha=0.1)

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

[x for x in df.columns if x.startswith('level2-Sustenance')]

df.columns

#target = 'pu_relative'
#target = 'median_house_value'
#target = 'count_pickup' # count_dropoff
target = 'count_dropoff'

# +
features = get_features(df, 2, log=True, winsorized=False, normalized=False)# + ['median_income']
#features = ['pca_1', 'pca_2']
#features = ['median_income']
#features = ['level2-Sustenance_log']

X = np.array(df.fillna(0)[features])
#poly = preprocessing.PolynomialFeatures(2)
#X = poly.fit_transform(X)
#features = poly.get_feature_names_out(features)



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


mae_avg = (df[target]-df[target].mean()).abs().mean()
print(f"MAE when predicting mean: {mae_avg}")
print()

reg = linear_model.LinearRegression().fit(X, Y)
print(f"r**2 of linear regression: {reg.score(X, Y)}")
mae_linreg = metrics.mean_absolute_error(Y, reg.predict(X))
print(f"MAE of linear regression: {mae_linreg}")
print(f"improvement over predicting mean: {1.0 - (mae_linreg/mae_avg)}")

print()
MAX_DEPTH = 2
reg2 = tree.DecisionTreeRegressor(max_depth=MAX_DEPTH).fit(X, Y)
print(f"r**2 for tree of max depth {MAX_DEPTH}: {reg2.score(X, Y)}")
mae_tree = metrics.mean_absolute_error(Y, reg2.predict(X))
print(f"MAE for tree of max depth {MAX_DEPTH}: {mae_tree}")
print(f"improvement over predicting mean: {1.0 - (mae_tree/mae_avg)}")


from sklearn import linear_model
reg3 = linear_model.LassoLarsCV().fit(X, Y)
print()
print(f"r**2 of lasso regression: {reg3.score(X, Y)}")
mae_lasso = metrics.mean_absolute_error(Y, reg3.predict(X))
print(f"MAE of lasso regression: {mae_lasso}")
print(f"improvement over predicting mean: {1.0 - (mae_lasso/mae_avg)}")


from sklearn import linear_model
reg4 = linear_model.Ridge(alpha=100).fit(X, Y)
mae_ridge = metrics.mean_absolute_error(Y, reg4.predict(X))
print()
print(f"r**2 of ridge regression: {reg4.score(X, Y)}")
print(f"MAE of ridge regression: {mae_ridge}")
print(f"improvement over predicting mean: {1.0 - mae_ridge/mae_avg}")


# +
# now doing it properly by using cross-validation instead of evaluating test performance. 

features = get_features(df, 2, log=True, winsorized=False, normalized=False)# + ['median_income']
X = np.array(df.fillna(0)[features])
y = df[target]

regressors = [
    dummy.DummyRegressor(strategy="mean"),
    linear_model.LinearRegression(),
    tree.DecisionTreeRegressor(max_depth=1),    
    tree.DecisionTreeRegressor(max_depth=2),
    tree.DecisionTreeRegressor(max_depth=3),    
    tree.DecisionTreeRegressor(max_depth=4),        
    tree.DecisionTreeRegressor(max_depth=5),
    tree.DecisionTreeRegressor(max_depth=6),        
    tree.DecisionTreeRegressor(max_depth=7),    
    tree.DecisionTreeRegressor(max_depth=10),    
    tree.DecisionTreeRegressor(max_depth=50),        
    tree.DecisionTreeRegressor(max_depth=100),            
    linear_model.LassoLarsCV(),
    linear_model.Ridge(alpha=100)
]

results = []

for regressor in regressors:
    
    regressor_pipeline = pipeline.make_pipeline(
        feature_selection.VarianceThreshold(), 
        preprocessing.StandardScaler(with_mean=False), 
        regressor)
    
    scorer = metrics.make_scorer(metrics.mean_absolute_error, greater_is_better=False)
    score = model_selection.cross_val_score(
        regressor_pipeline, X, y, cv=5, scoring=scorer
    ).mean()

    results.append([str(regressor), score])
    
results
# -

[(a, 1.0-(b/results[0][1])) for a,b in results]

_[3:-2]

[(a.split('=')[1][:-1], b) for a, b in _]

df_results = pd.DataFrame(_, columns=['max_depth', 'improvement'])

df_results

sns.barplot(data=df_results, x="max_depth", y="improvement", color="blue")
sns.despine(right=True, top=True)

# +
# below: more experiments that didn't make it into the talk
# -

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X)

pca.transform(X).T

df_pca = pd.DataFrame(pca.transform(X), columns=['x1','x2'])
df_pca.plot.scatter(x='x1', y='x2', alpha=0.1)

df['pca_1'] = pca.transform(X).T[0]
df['pca_2'] = pca.transform(X).T[1]

pca.components_

sns.regplot(data=df, x='pca_1', y='count_pickup')

df['count_pickup_log'] = np.log(df['count_pickup'])

sns.scatterplot(data=df.sort_values('count_pickup_log', ascending=True), x='pca_1', y='pca_2', hue='count_pickup_log', alpha=0.2, palette='coolwarm')

pd.DataFrame(df.iloc[2][features]).reset_index().plot.bar(x='index')

df.sort_values('count_dropoff', ascending=False)[['index', 'count_dropoff'] + features ][-100:]

index=268
row = df[df['index']==index].iloc[0][features]
pd.DataFrame(row).reset_index().plot.bar(x='index')

len(df)

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

sns.kdeplot(x=df['pca_1'], y=df['count_dropoff'])

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

# +
from sklearn.tree import _tree

def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    print("def tree({}):".format(", ".join(feature_names)))

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print("{}if {} <= {}:".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            print("{}else: if {} > {}:".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
        else:
            print("{}return {}".format(indent, tree_.value[node]))

    recurse(0, 1)


# -

tree_to_code(reg2, feature_names=features)

from sklearn.tree import export_text
print(export_text(reg2, feature_names=features))

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

df

# +
# create a compact lat/lon-projected GeoJSON file for kepler.gl
# -

df_ll = df.to_crs(4326)

[f for f in df_ll.columns if not f.startswith('level')]

df_ll = df_ll[['count_pickup', 'count_dropoff', 'geometry']]

df_ll.to_file(LATLON_OUTPUT_FILE, driver='GeoJSON')

df_ll


