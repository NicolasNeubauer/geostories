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

# Load taxi zones, ride counts, POIs, and merge them, then save the result into `OUTPUT_FILE`

# +
from collections import Counter

import pandas as pd, geopandas as gpd, json
import seaborn as sns
import matplotlib.pyplot as plt
# -

DATA_DIR = '/Users/nicolasneubauer/Data/geostories'       # set to your own
POI_FILE = f'{DATA_DIR}/POIs_NYC.geojson'                 # create via read_osm.py
TAXI_ZONES_FILE = f"{DATA_DIR}/taxi_zones/taxi_zones.shp" # download from https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page 
JAN20_FILE = f"{DATA_DIR}/yellow_tripdata_2020-01.csv"    # see above
JAN21_FILE = f"{DATA_DIR}/yellow_tripdata_2021-01.csv"    # see above
OUTPUT_FILE = f"{DATA_DIR}/full_dataset_NYC.geojson" 

# ## Taxi zones

gdf_taxi = gpd.read_file(TAXI_ZONES_FILE)) # 
gdf_taxi.columns

gdf_taxi.crs

str(gdf_taxi['geometry'][0])[0:200] # see custom projection

gdf_taxi = gdf_taxi.to_crs('epsg:4326')
str(gdf_taxi['geometry'][0])[0:200] # proper lat/lon

gdf_taxi.plot()

# ## Trip data

df_jan20 = pd.read_csv(JAN20_FILE, low_memory=False)
df_jan20.head()

df_jan21 = pd.read_csv(JAN21_FILE, low_memory=False)
df_jan21.head()

# ## merging trip data to taxi zones

# +
IDCOLUMN="OBJECTID" # ID of taxi zones

def count(input_df, by_column, output_column):
    """Aggregates trips by pickup or dropoff location (defined by `by_column`). `output_column` is variable to
       describe type (pickup vs dropoff) and time."""
    return input_df\
        .groupby(by_column)\
        .count()[['VendorID']]\
        .reset_index()\
        .rename(columns={'VendorID': output_column, by_column: IDCOLUMN})

df_pu_20 = count(df_jan20, 'PULocationID', 'num_pickups_20')
df_pu_21 = count(df_jan21, 'PULocationID', 'num_pickups_21')
df_do_20 = count(df_jan20, 'DOLocationID', 'num_dropoffs_20')
df_do_21 = count(df_jan21, 'DOLocationID', 'num_dropoffs_21')

df_do_21

# +
gdf_taxi_count = gdf_taxi\
    .merge(df_pu_20, how='left', on=IDCOLUMN)\
    .merge(df_pu_21, how='left', on=IDCOLUMN)\
    .merge(df_do_20, how='left', on=IDCOLUMN)\
    .merge(df_do_21, how='left', on=IDCOLUMN).fillna(0)

gdf_taxi_count.drop(columns=['geometry']) 
# -

gdf_taxi_count[gdf_taxi_count['num_dropoffs_20'] == 0].drop(columns=['geometry'])

gdf_taxi_count = gdf_taxi_count.loc[(gdf_taxi_count['num_dropoffs_20'] > 0) & (gdf_taxi_count['num_pickups_20'] > 0)]

gdf_taxi_count['do_relative'] = gdf_taxi_count['num_dropoffs_21']/gdf_taxi_count['num_dropoffs_20']

gdf_taxi_count['pu_relative'] = gdf_taxi_count['num_pickups_21']/gdf_taxi_count['num_pickups_20']

gdf_taxi_count.plot.scatter(x='num_pickups_20', y='num_pickups_21')

gdf_taxi_count['num_pickups_20'].hist(cumulative=True, density=1, bins=100)

# normalize by area
# we don't really care later on because we want to predict 2021 / 2020 relationship which factors out area anyhow 
for field in ['pickups', 'dropoffs']:
    for year in [20, 21]:
        org_fieldname = f"num_{field}_{year}"
        new_fieldname = f"{org_fieldname}_normalized"
        gdf_taxi_count[new_fieldname] = gdf_taxi_count[org_fieldname] / gdf_taxi_count['Shape_Area']

gdf_taxi_count[['zone','num_pickups_20','num_pickups_21','num_dropoffs_20', 'num_dropoffs_21', 'do_relative', 'pu_relative']].head()

# ## Extract relevant POIs

# ### inspect

gdf_poi = gpd.read_file(POI_FILE)

ax = gdf_poi.plot()
gdf_taxi.plot(ax=ax, color='red')

ax = gdf_poi[gdf_poi['level3']=='level3-amenity-parking'].plot(alpha=0.01)
gdf_taxi.plot(ax=ax, color='red', alpha=0.1)

# +
df_count=gpd.sjoin(gdf_taxi, gdf_poi).\
  groupby([IDCOLUMN, 'level1', 'level2', 'level3']).\
  agg('count')[["Shape_Leng"]].\
  rename(columns={'Shape_Leng': 'count'})

df_count
# -

ax = gdf_taxi[gdf_taxi['OBJECTID']==4].plot(color='white', edgecolor='grey')
gpd.sjoin(gdf_poi, gdf_taxi[gdf_taxi['OBJECTID']==4]).plot(ax=ax)

import contextily 
ax = gpd.sjoin(gdf_poi, gdf_taxi[gdf_taxi['OBJECTID']==4])\
  .set_crs(epsg=4326)\
  .to_crs(epsg=3857)\
  .plot(figsize=(10, 7), column="level2")
               
    #.plot(color='white', edgecolor='grey')\
gdf_taxi[gdf_taxi['OBJECTID']==4]\
  .set_crs(epsg=4326)\
  .to_crs(epsg=3857)\
  .plot(ax=ax, facecolor='blue', edgecolor='black', alpha=0.1)
contextily.add_basemap(
    ax, source=contextily.providers.OpenStreetMap.BZH)

gdf_poi

df_count['count'].sum()

# ### aggregate by level1

df_tags_per_object = df_count.\
  reset_index().\
  drop(columns=['level2', 'level3']).\
  groupby([IDCOLUMN, 'level1']).\
  sum().\
  reset_index()
df_tags_per_object

df_tags_per_object['count'].sum()

df_tags_per_object = pd.\
  pivot(df_tags_per_object, index=IDCOLUMN, columns='level1', values='count')\
  .fillna(0)
df_tags_per_object

df_tags_per_object.sum()

gdf_taxi_count_with_tag_count = gdf_taxi_count.join(df_tags_per_object)

# ### now the same but for level3 tags

df_values_per_object = df_count.\
  reset_index().\
  drop(columns=['level1', 'level2']).\
  groupby([IDCOLUMN, 'level3']).\
  sum().\
  reset_index()
df_values_per_object = pd.\
  pivot(df_values_per_object, index=IDCOLUMN, columns='level3', values='count').\
  fillna(0)
df_values_per_object

df_tag_value_count = pd.DataFrame(df_values_per_object.sum().reset_index())
df_tag_value_count.sort_values(0, ascending=False)[0:30]

gdf_taxi_count_with_tag_count = gdf_taxi_count_with_tag_count.join(df_values_per_object)

sns.regplot(data=gdf_taxi_count_with_tag_count[gdf_taxi_count['num_dropoffs_20']>10000], 
            x='level3-amenity-restaurant', 
            y='num_dropoffs_20')

# ### finally, aggregate by the newly created level2 (grouping amenity types into higher-level categories)

df_values_per_object = df_count.\
  reset_index().\
  drop(columns=['level1', 'level3']).\
  groupby([IDCOLUMN, 'level2']).\
  sum().\
  reset_index()
df_values_per_object = pd.\
  pivot(df_values_per_object, index=IDCOLUMN, columns='level2', values='count').\
  fillna(0)
df_values_per_object

gdf_taxi_count_with_tag_count = gdf_taxi_count_with_tag_count.join(df_values_per_object)

# ### visualize and save

gdf_taxi_count_with_tag_count.head()

l2 = [c for c in gdf_taxi_count_with_tag_count.columns if c.startswith('level2')]

gdf_taxi_count_with_tag_count[["zone", "num_pickups_20", "do_relative"] + l2].fillna(0).head()

gdf_taxi_count_with_tag_count.to_file(OUTPUT_FILE, driver='GeoJSON')

import keplergl

keplergl.KeplerGl(height=1000, data={'data': gdf_taxi_count_with_tag_count})
