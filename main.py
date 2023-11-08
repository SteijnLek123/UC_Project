# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import osmnx as ox
import matplotlib as plt
import geopandas as gpd


def main():
    region_name = 'Oklahoma City'
    area = ox.geocode_to_gdf(region_name)
    print(area.head())

if __name__ == '__main__':
    main()

