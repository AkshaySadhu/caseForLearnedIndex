import xml.etree.ElementTree as ET
import time
from sortedcontainers import SortedList
import pandas as pd

tree = ET.parse('sydney2.osm')
root = tree.getroot()

longitudes = []
latitudes = []

for node in root.findall('node'):
    lon = node.get('lon')
    lat = node.get('lat')
    if lon and lat:
        longitudes.append(lon)
        latitudes.append(lat)

data = {
        #'longitude': longitudes,
        'latitude': list(set(latitudes)),
        'index': list(range(len(set(latitudes))))
       }

df = pd.DataFrame(data)
df.to_csv('sydneyUniqueUnsortedLatitudes.csv', index=False)



print("Total tuples: ", len(longitudes))
# print(len(latitudes))

data = {
        #'longitude': longitudes,
        'latitude': latitudes,
        'index': list(range(len(latitudes)))
       }
df = pd.DataFrame(data)
df.to_csv('sydneyAllLatitudes.csv', index=False)

print(len(data['latitude']))


print(len(data['latitude']))

data = {
        #'longitude': longitudes,
        'latitude': list(sorted(set(latitudes))),
        'index': list(range(len(set(latitudes))))
       }

df = pd.DataFrame(data)
df.to_csv('sydneyUniqueSortedLatitudes.csv', index=False)

print(len(data['latitude']))







unique_lat = len(set(latitudes))
unique_long = len(set(longitudes))

print("Duplicate latitudes: ", len(latitudes)-unique_lat)


# print("Duplicate longitudes: ", len(longitudes)-unique_long)

# bTree = SortedList()
# for lon in longitudes:
#     bTree.add(float(lon))

# minLat = 151.271
# maxLat = 151.279
# start_time = time.perf_counter()  # Start timer
# results = bTree.irange(minLat, maxLat)
# end_time = time.perf_counter()


# # print(f"Nodes with latitude between {minLat} and {maxLat}:")
# # for long in results:
# #     print(f"Latitude: {long}")
# elapsed_time = end_time - start_time
# print(f"Time taken for query: {elapsed_time:.6f} seconds")
