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

# This notebooks creates a GeoJSON file (defined in `POI_FILE`) which contains all POIs for a city of a given tag.

# +
from collections import defaultdict, Counter

import pyrosm # for downloading only, unfortunately processing crashes
import osmium as o
import shapely.wkb as wkblib
# #!pip install beautifulsoup4 
from bs4 import BeautifulSoup

import pandas as pd, geopandas as gpd
# -

DATA_DIR = '/Users/nneubaue/Data/osm'
AMENITIES_FILE = f"{DATA_DIR}/amenities.html" # https://wiki.openstreetmap.org/wiki/Template:Map_Features:amenity

REGION = 'newyorkcity' # ID for pyrosm to download from OSM
POI_FILE = f'{DATA_DIR}/POIs_NYC.geojson' # output file

REGION = 'northern_california'
POI_FILE = f'{DATA_DIR}/POIs_northern_california.geojson'

REGION = 'southern_california'
POI_FILE = f'{DATA_DIR}/POIs_southern_california.geojson'


def get_level2_map(filename):
    """returns a dictionary that maps tags to an intermediate 'level2' category for amenities"""
    with open(filename) as fp:
        soup = BeautifulSoup(fp, "html.parser")    
    table = soup.body.find_all("table")[0]

    to_level2 = {}
    for tr in table.find_all('tr')[1:]: # ignore header
        h4 = tr.find('h4')
        if h4:
            lvl2 = h4.find('span')['id']
            if lvl2.startswith("Entertainment"):
                lvl2 = "Entertainment" # shorten very long tag
            continue
        lvl3 = tr['id']
        to_level2[lvl3] = lvl2
        
    return to_level2


# +
wkbfab = o.geom.WKBFactory()
i = 0

class TagListHandler(o.SimpleHandler):
    """adapted from https://github.com/osmcode/pyosmium/blob/master/examples/amenity_list.py"""
    
    def __init__(self, tags, output_list, level2_map):
        o.SimpleHandler.__init__(self)
        self.tags = tags
        self.output_list = output_list
        self.level2_map = level2_map
        
    def level2(self, level1, level3):
        if level3.startswith('amenity') and level3 not in self.level2_map:
            print(level3)
        return self.level2_map.get(level3, level1)
        
    def store_tag(self, tags, tag, lon, lat):
        global i
        if i % 1000 == 0:
            print(i)
        i += 1
        lvl1 = f'level1-{tag}'
        lvl3_name = f"{tag}-{tags[tag]}"
        lvl2 = f"level2-{self.level2(tag, lvl3_name)}"
        lvl3 = f"level3-{lvl3_name}"
        self.output_list.append((lat, lon, lvl1, lvl2, lvl3))
 
    def node(self, n):
        for tag in self.tags:
            if tag in n.tags:
                self.store_tag(n.tags, tag, n.location.lon, n.location.lat)
                
                # only store one tag so we're counting each location max once
                return 

    def area(self, a):
        for tag in self.tags:
            if tag in a.tags:
                try:
                    wkb = wkbfab.create_multipolygon(a)
                except Exception as e:
                    print(e)
                    continue                    
                poly = wkblib.loads(wkb, hex=True)
                centroid = poly.representative_point()
                self.store_tag(a.tags, tag, centroid.x, centroid.y)
                
                # only store one tag so we're counting each location max once
                return

            
tags = ['amenity', 'shop', 'leisure', 'tourism', 'office']

def process(osmfile, tags=tags, level2_map={}):
    global i
    i = 0
    output = []
    handler = TagListHandler(tags, output, level2_map=level2_map)
    handler.apply_file(osmfile)
    return output


# -

pbf_file = pyrosm.get_data(REGION, directory=DATA_DIR)

level2_map = get_level2_map(AMENITIES_FILE)

output = process(pbf_file, level2_map=level2_map)

l = []
for (lat, lon, lvl1, lvl2, lvl3) in output:
    l.append(lvl2)
Counter(l).most_common(20)

df_poi = pd.DataFrame(output, columns=['lat', 'lon', 'level1', 'level2', 'level3'])
df_poi.head()

gdf_poi = gpd.GeoDataFrame(df_poi, geometry=gpd.points_from_xy(df_poi['lon'], df_poi['lat'])).set_crs('EPSG:4326')
gdf_poi.plot()

gdf_poi.to_file(POI_FILE, driver='GeoJSON')

len(gdf_poi)

POI_FILE






