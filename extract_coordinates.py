#! Python3
# Calls API from open street map (OSM) and takes lat and long from search query
# if the coordinates cannot be found in OSM it will look for it in onemap.sg
# if it cant be found there as well, then it will be filled with 0 for lat and lon

# Resources
# source : https://nominatim.org/release-docs/latest/api/Search/
# https://medium.com/@adri.espejo/getting-started-with-openstreetmap-nominatim-api-e0da5a95fc8a
# https://www.onemap.gov.sg/docs/
# https://github.com/mylee16/onemap-api

import pandas as pd
import numpy as np
import requests

hdb_string = './resale-flat-prices-based-on-registration-date-from-jan-2017-onwards.csv'

df = pd.read_csv(hdb_string)

df['Address'] = df['block'] + " " + df['street_name']
# API only accepts certain format so having them fed into the API as 'Block number' 'Street Name ' works best
addresslist = list(df['Address'])


all_coordinates = []

for index, address in enumerate(addresslist):
    print('Progress {} percent \n'.format(np.round(index/len(addresslist)*100,2)))
    print('Searching for ', address)
    query = 'https://nominatim.openstreetmap.org/search?q=' + address + '&countrycodes=sg&limit=1&format=json'
    res = requests.get(query)
    result = eval(res.text)
    try:
        lat = result[0]['lat']
        long = result[0]['lon']
        print('Lat is ', lat)
        print('Long is ' , long)
        all_coordinates.append([lat, long])
        print('Information Found and appended \n')
        print('*'*25 , '\n')
    except:
        try:
            print('No results from openstreetmap, searching from onemap.sg')
            second_query = 'https://developers.onemap.sg/commonapi/search?searchVal='+ address +'&returnGeom=Y&getAddrDetails=Y&pageNum=1'
            res = requests.get(second_query)
            result = eval(res.text)
            lat = result['results'][0]['LATITUDE']
            lon = result['results'][0]['LONGITUDE']
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
df_combined  = df_combined.rename(columns={0:'Latitude', 1:'Longitude'})

df_combined.to_csv('resalehdb_data_withCoordinates.csv',index=False)