from typing import Optional
from pprint import pprint, pformat
from datetime import datetime
import pandas as pd
from hdb_mrt.mrt_onemap import return_geo_one_address, return_mrt_with_geo , return_closest_mrt_distances, extract_mrt_excelsheet
from app.api.config import SETTINGS

# testing
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sample_dict = {
    'postal_code': 560325,
    'storey_range':'01 TO 03',
    'flat_type':'4 ROOM'
    }

datapath = 'data/hdb_latest.csv'

dataset = pd.read_csv(datapath)

def fetch_resources(input:dict)->Optional[pd.DataFrame]:
    """
    takes the input from the user and returns the information available 
    from the dataset
    information to be returned:
    - Flat model
    - Floor Area - sqm
    - Remaining lease
    - lat
    - long
    - nearest mrt
    - distance to nearest mrt
    - month
    """

    # fetch address from postal code
    address = return_geo_one_address(one_map_url= SETTINGS.dict()['ONEMAP_API'], query= input['postal_code'])

    mrt_file = extract_mrt_excelsheet(SETTINGS.dict()['MRT_API'],SETTINGS.dict()['MRT_EXCEL_FILE'])
    mrt_stations_geo = return_mrt_with_geo(SETTINGS.dict()['ONEMAP_API'],mrt_file)

    # format into pd DataFrame
    query = pd.DataFrame.from_dict(address, orient='index').T

    res = return_closest_mrt_distances(query.reset_index(), mrt_stations_geo)
    columns = ['index','full_address', 'lat', 'long', 'nearest_mrt', 'nearest_distance_to_mrt']
    res.columns = (columns)
    lat = res.lat.values[0]
    long = res.long.values[0]
    nearest_mrt = res.nearest_mrt.values[0]
    nearest_distance_to_mrt = res.nearest_distance_to_mrt.values[0]
    full_address = res.full_address.values[0]

    dataset = pd.read_csv(SETTINGS.dict()['DATA_PATH'])

    # only need the first item
    sample_data = dataset[(dataset.lat == lat) & (dataset.long == long) & (dataset.month.str.contains("20")) ].iloc[0,:]

    # information not found in dataset i.e new address, invalid postal code
    if len(sample_data) == 0:
        return None

    else:
        # load data from dataset

        town = sample_data.town
        flat_model = sample_data.flat_model
        address = sample_data.address
        floor_area_sqm = sample_data.floor_area_sqm
        block = sample_data.block
        year = datetime.now().year
        month = f"{datetime.now():%Y-%m}"
        street_name = sample_data.street_name
        lease_commence_date = sample_data.lease_commence_date

        complete_res = {
            'town':town,
            'flat_type': input['flat_type'],
            'flat_model': flat_model,
            'floor_area_sqm': floor_area_sqm,
            'street_name': street_name,
            'resale_price': 0,
            'month': month,
            'lease_commence_date': lease_commence_date,
            'storey_range': input['storey_range'],
            'block': block, # sometimes the block is not in the full address or postal code
            'remaining_lease': year - lease_commence_date,
            'address': address,
            'full_address': full_address,
            'lat': lat,
            'long': long,
            'nearest_mrt': nearest_mrt,
            "nearest_distance_to_mrt": nearest_distance_to_mrt
        }

        logger.info(pformat(complete_res))

        complete_query = pd.DataFrame.from_dict(complete_res, orient='index').T

        return complete_query

    # # previous code 
    # resources = input
    # month = f"{datetime.now().year}-{datetime.now().month}"
    # resources['month'] = month

    # address = return_geo_one_address(settings['ONEMAP_API'], input['postal_code'])

    # logger.info(f"Result returned : {pformat(address)}")

    # dataset = dataset[~dataset.duplicated(subset=['town','flat_type','flat_model','storey_range','floor_area_sqm','address']) \
    #     & (dataset.month.str.contains("20"))]

    # subset = dataset[(dataset['street_name']== input['street_name']) & \
    #     (dataset['block']== input['block'])& \
    #         (dataset.flat_type == input['flat_type']) & \
    #             (dataset.storey_range == input['storey_range'])]

    # logger.info(pformat(subset))

    # # not unique
    # if len(subset) != 1:
    #     resources['valid'] = 'False'
    #     return resources

    # else:
    #     resources['town'] = subset['town'].values[0]
    #     resources['flat_model'] = subset['flat_model'].values[0]
    #     resources['floor_area_sqm'] = subset['floor_area_sqm'].values[0]
    #     resources['lease_commence_date'] = subset['lease_commence_date'].values[0]
    #     resources['remaining_lease'] = int(datetime.now().year) - int(resources['lease_commence_date'])
    #     resources['address'] = subset['address'].values[0]
    #     resources['full_address'] = subset['full_address'].values[0]
    #     resources['lat'] = subset['lat'].values[0]
    #     resources['long'] = subset['long'].values[0]
    #     resources['nearest_mrt'] = subset['nearest_mrt'].values[0]
    #     resources['nearest_distance_to_mrt'] = subset['nearest_distance_to_mrt'].values[0]
    #     resources['valid'] = 'True'
    #     return resources

def validate_resources(resources:dict, dataset:pd.DataFrame)->dict:
    """
    validates the output of fetch resources.
    Say if the information fetched from dataset is not available
    does not mean that the address is invalid, it could be that
    the address is new i.e yet to be sold so it does not exist
    in the dataset, thus the information will need to be retrieved

    """
    if resources['valid'] == 'True':
        return resources

    # no information found
    else:
        # basically if it is not found within the dataset, all of the 
        # information needs to be generated and returned
        dataset


if __name__ == '__main__':
    res = fetch_resources(sample_dict, dataset)
    pprint(res)