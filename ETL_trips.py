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

import numpy as np, pandas as gpd, geopandas as gpd
import matplotlib.pyplot as plt
import shapely

DATA_DIR = '/Users/nneubaue/Data/osm' 
INPUT_FILE = f"{DATA_DIR}/nyc-taxi-trip-duration/train.csv"
POI_FILE = f'{DATA_DIR}/POIs_NYC.geojson'
OUTPUT_FILE = f"{DATA_DIR}/full_dataset_trips_NYC.geojson"

CRS = 'epsg:2263' 

gdf = gpd.read_file(INPUT_FILE)
len(gdf)


def get_gdf(gdf, field, crs=CRS):
    otherfield = 'dropoff' if field == 'pickup' else 'pickup'
    return gdf\
        .set_geometry(gpd.points_from_xy(
            gdf[f'{field}_longitude'], 
            gdf[f'{field}_latitude']))\
        .set_crs('epsg:4326')\
        .to_crs(crs)\
        .drop(columns=[
            f'{field}_longitude', 
            f'{field}_latitude', 
            f'{otherfield}_longitude', 
            f'{otherfield}_latitude'])


gdf_pickup = get_gdf(gdf, 'pickup')
gdf_pickup

gdf_dropoff = get_gdf(gdf, 'dropoff')
gdf_dropoff

gdf_pickup.iloc[0:10000].plot()

# +
# efficient gridding via https://james-brennan.github.io/posts/fast_gridding_geopandas/

# total area for the grid
xmin, ymin, xmax, ymax = 920000, 160000, 1060000, 260000 # gdf_pickup.total_bounds
xmin, ymin, xmax, ymax = 970000, 190000, 1010000, 250000 # gdf_pickup.total_bounds

print(xmin, ymin, xmax, ymax)
# how many cells across and down
n_cells=100
cell_size_x = (xmax-xmin)/n_cells
cell_size_y = (ymax-ymin)/n_cells
# create the cells in a loop
grid_cells = []
for x0 in np.arange(xmin, xmax+cell_size_x, cell_size_x):
    for y0 in np.arange(ymin, ymax+cell_size_y, cell_size_y):
        # bounds
        x1 = x0-cell_size_x
        y1 = y0+cell_size_y
        grid_cells.append(shapely.geometry.box(x0, y0, x1, y1))
gdf_cell = gpd.GeoDataFrame(grid_cells, columns=['geometry'], crs=CRS)
# -

gdf_cell.plot(facecolor='none', edgecolor='grey')

ax = gdf_cell.plot(facecolor="none", figsize=(12, 8), edgecolor='grey', alpha=0.1)
gdf_pickup.iloc[0:1000].plot(markersize=.1, ax=ax)
#plt.autoscale(False)
#ax.axis("off")

gdf_joint = gpd.sjoin(gdf_cell.reset_index(), gdf_pickup)



gdf_count = gdf_joint.groupby('index').count()[['geometry']].reset_index().rename(columns={'geometry': 'count'})
gdf_count = gdf_count.set_index('index').join(gdf_cell)

gdf_count.plot(column='count')

gdf_poi = gpd.read_file(POI_FILE)

gdf_poi = gdf_poi.set_crs('epsg:4326').to_crs(CRS)

gdf_count = gdf_count.set_crs(CRS)

# +
df_count=gpd.sjoin(gdf_count, gdf_poi).\
  groupby(['index', 'level1', 'level2', 'level3']).\
  agg('count')[['count']].reset_index()

df_count
# -

df_count.sort_values('count', ascending=False)

i = 1140
gdf_count_2 = gdf_count.reset_index()
gdf_selected = gdf_count_2[gdf_count_2['index']==i]
ax = gdf_selected.plot(color='white', edgecolor='grey')
gpd.sjoin(gdf_poi, gdf_selected).plot(ax=ax)

gpd.sjoin(gdf_poi, gdf_selected)

gdf_count

IDCOLUMN="index"

df_tags_per_object = df_count.\
  drop(columns=['level2', 'level3']).\
  groupby([IDCOLUMN, 'level1']).\
  sum().\
  reset_index()
df_tags_per_object

df_tags_per_object = pd.\
  pivot(df_tags_per_object, index=IDCOLUMN, columns='level1', values='count')\
  .fillna(0)
df_tags_per_object

gdf_final = gdf_count.join(df_tags_per_object)
gdf_final

df_values_per_object = df_count.\
  drop(columns=['level1', 'level2']).\
  groupby([IDCOLUMN, 'level3']).\
  sum().\
  reset_index()
df_values_per_object = pd.\
  pivot(df_values_per_object, index=IDCOLUMN, columns='level3', values='count').\
  fillna(0)
df_values_per_object

gdf_final = gdf_final.join(df_values_per_object)
gdf_final


df_tags_per_object

df_values_per_object = df_count.\
  drop(columns=['level1', 'level3']).\
  groupby([IDCOLUMN, 'level2']).\
  sum().\
  reset_index()
df_values_per_object = pd.\
  pivot(df_values_per_object, index=IDCOLUMN, columns='level2', values='count').\
  fillna(0)
df_values_per_object

gdf_final = gdf_final.join(df_values_per_object)
gdf_final


gdf_final = gdf_final.fillna(0)
gdf_final

len(gdf_final)

gdf_final.plot()

gdf_final.to_file(OUTPUT_FILE, driver='GeoJSON')

gdf_joint_dropoff = gpd.sjoin(gdf_cell.reset_index(), gdf_dropoff)
gdf_joint_dropoff

gdf_count_dropoff = gdf_joint_dropoff.groupby('index').count()[['geometry']].reset_index().rename(columns={'geometry': 'count_dropoff'})
gdf_count_dropoff = gdf_count_dropoff.set_index('index').join(gdf_cell)
gdf_count_dropoff

gdf_final2 = gdf_final.join(gdf_count_dropoff.drop(columns=['geometry']), how='inner').rename(columns={'count': 'count_pickup'})
gdf_final2

gdf_final2 = gdf_final2.fillna(0)



gdf_final2[0:1000].plot()

gdf_final2.to_file(OUTPUT_FILE, driver='GeoJSON')

gdf_poi[gdf_poi['level2']=='level2-Facilities']['level3'].unique()

gdf_poi[gdf_poi['level2']=='level2-leisure']['level3'].unique()


