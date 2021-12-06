#! Python3
# Calls API from open street map (OSM) and takes lat and long from search query

# Resources
# source : https://nominatim.org/release-docs/latest/api/Search/
# https://medium.com/@adri.espejo/getting-started-with-openstreetmap-nominatim-api-e0da5a95fc8a

import pandas as pd
import numpy as np
import requests

hdb_string = './resale-flat-prices-based-on-registration-date-from-jan-2017-onwards.csv'

df = pd.read_csv(hdb_string)

df['Address'] = df['block'] + " " + df['street_name']
# Prints only every 20 entries, This reduces total dataset of 113959 to a much manageable number
# API only accepts certain format so having them fed into the API as 'Block number' 'Street Name ' works best
addresslist = list(df['Address'])


all_coordinates = []

for index, address in enumerate(addresslist):
    search_query = 'https://nominatim.openstreetmap.org/search?q=' + address + '&countrycodes=sg&limit=1&format=json'
    print('\n','*'*25)
    print('Searching for address ' + address + '\n')
    print('Progress {} percent \n'.format(np.round(index/len(addresslist),2)))
    res = requests.get(search_query)
    search_result = eval(res.text)
    try:
        lat = search_result[0]['lat']
        long = search_result[0]['lon']
        print('Lat is ', lat)
        print('Long is ' , long)
        all_coordinates.append([lat, long])
        print('Information Found and appended \n')
        print('*'*25 , '\n')
    except:
        print('Lat and long is unavailable')
        all_coordinates.append([0, 0])
        print('Information missing and appended null values \n')
        print('*'*25 , '\n')
        continue

df_coordinates = pd.DataFrame(all_coordinates)
df_combined = df.join(df_coordinates)
df_combined  = df_combined .rename(columns={0:'Latitude', 1:'Longitude'})

df_combined.to_csv('resalehdb_data_withCoordinates.csv',index=False)