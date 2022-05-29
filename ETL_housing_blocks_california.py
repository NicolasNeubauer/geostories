# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import matplotlib.pyplot as plt
import pandas as pd, geopandas as gpd

DATA_DIR = "/Users/nneubaue/Data/osm"
SHAPEFILE = f"{DATA_DIR}/tl_2021_06_bg/tl_2021_06_bg.shp"
DATAFILE = f"{DATA_DIR}/housing.csv"
POI_FILE_1 = f'{DATA_DIR}/POIs_northern_california.geojson'
POI_FILE_2 = f'{DATA_DIR}/POIs_southern_california.geojson'
OUTPUT_FILE = f"{DATA_DIR}/full_dataset_california.geojson"

gdf = gpd.read_file(shapefile)

gdf.plot()

len(gdf)

gdf.drop(columns=['geometry']).head()

df = pd.read_csv(DATAFILE)

df.head()

len(df)

gdf["NAMELSAD"].value_counts()

for key, dfx in gdf.groupby("NAMELSAD"):
    print(key)
    dfx.plot()

gdf_data = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["longitude"], df["latitude"]))

gdf_data.plot()

gdf = gdf.to_crs("EPSG:4326")

gdf_blocks_with_data = gpd.sjoin(gdf, gdf_data.set_crs('EPSG:4326'), how="inner").rename(columns={'index_right': 'id'})

len(gdf_blocks_with_data)

gdf_blocks_with_data.plot(column="median_house_value")
plt.gcf().set_size_inches(18.5, 10.5)





gdf_pois = pd.concat([gpd.read_file(f) for f in (POI_FILE_1, POI_FILE_2)])

gdf_pois

len(gdf_pois)

type(gdf_pois)

IDCOLUMN="id"
col = gdf_blocks_with_data.columns[0]
df_count = gpd.sjoin(gdf_blocks_with_data, gdf_pois, how="inner").\
  groupby([IDCOLUMN, 'level1', 'level2', 'level3']).\
  agg('count')[[col]].\
  rename(columns={col: 'count'})
df_count


df_tags_per_object = df_count.\
  reset_index().\
  drop(columns=['level2', 'level3']).\
  groupby([IDCOLUMN, 'level1']).\
  sum().\
  reset_index()
df_tags_per_object

df_tags_per_object = pd.\
  pivot(df_tags_per_object, index=IDCOLUMN, columns='level1', values='count')\
  .fillna(0)
df_tags_per_object

gdf_all = gdf_blocks_with_data.join(df_tags_per_object)

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

gdf_all = gdf_all.join(df_values_per_object)

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

gdf_all = gdf_all.join(df_values_per_object)

gdf_all.to_file(OUTPUT_FILE, driver="GeoJSON")


