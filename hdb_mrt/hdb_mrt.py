"""This package is meant to be part of a pipeline to extract the 
information entirely from data.gov.sg and the MRT information from 
lta's datamall

The relevant closest MRT information will be appended onto the 
resale HDB data and returned
"""
import pandas as pd
import hydra
import requests
from typing import List
import requests.packages.urllib3.util.connection as urllib3_cn
import os
import json

import hdb
import mrt_onemap
import logging


# logger = logging.getLogger(__name__)


# def setup_logging(logging_config_path:str,
#                 default_level=logging.INFO):
#     """Set up configuration for logging utilities.

#     Parameters
#     ----------
#     logging_config_path : str, optional
#         Path to YAML file containing configuration for Python logger
#     default_level : logging object, optional, by default logging.INFO
#     """

#     try:
#         with open(logging_config_path, "rt") as file:
#             log_config = json.load(file)
#             logging.config.dictConfig(log_config)

#     except Exception as error:
#         logging.basicConfig(
#             format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
#             level=default_level)
#         logger.info(error)
#         logger.info(
#             "Logging config file is not found. Basic config is being used.")

@hydra.main(config_path="../conf/", config_name="config.yaml")
def main(args):
    """
    Main function to generate the data in excel file
    """
    
    # logger = logging.getLogger(__name__)
    # logger.info("Setting up logging configuration.")
    # logger_config_path = os.path.\
    #     join(hydra.utils.get_original_cwd(),
    #         "conf","logging.yml")
    # setup_logging(logger_config_path)

    ###########################################
    ####### Initialisation of Variables #######
    ###########################################

    limit = args['specs']['limit']
    data_gov_api = args['specs']['data_gov_api']
    write_file_name = args['specs']['write_file_name']
    onemap_api = args['specs']['onemap_api']
    mrt_api = args['specs']['mrt_api']
    mrt_excel_file = args['specs']['mrt_excel_file']

    resource_post2017 = args['resource_mapping']['post2017']
    resource_jan2015todec2016 = args['resource_mapping']['jan2015todec2016']
    resource_mar2012todec2014 = args['resource_mapping']['mar2012todec2014']
    resource_jan2000tofeb2012 = args['resource_mapping']['jan2000tofeb2012']
    resource_pre1999 = args['resource_mapping']['pre1999']

    urls = [
        resource_pre1999,
        resource_jan2000tofeb2012, 
        resource_mar2012todec2014,
        resource_jan2015todec2016, 
        resource_post2017 
        ]

    # force ipv4
    urllib3_cn.allowed_gai_family = mrt_onemap.allowed_gai_family

    ###########################################
    #######         Read Data           #######
    ###########################################

    """
    Main files: 
    - unique_address_geo_valid_mrt
        - all HDB records then take unique address
        - + geo
        - + remove invalid
        - split into Pre and Post 2000
        - + append mrt with nearest

    - Invalid Address 
        - unique_address_geo with no geo data

    - HDB 
        - Post and Pre 2000 files
        - Drop observations with invalid addresses
    
    - Final File
        - Append HDB files with correct unique_address_geo_valid_mrt
        - Combine together

    Unique address needs to be generated first by gathering all of the 
    unique address in the HDB dataset, then we need to validate
    the addresses by retrieving its geo properties from onemap.sg

    MRT stations will need to be pulled from the LTA api and its
    geo properties retrieved from onemap.sg
    The MRT stations will be further split into two segments, one pre 2000
    and one post 2000 for distance measurements to HDBs from existing MRTs
    (a MRT that has yet to be built in 2020 cannot be used against a HDB 
    observation in the 2000)

    In order to prevent additional calculation, only the unique addresses
    will have their distances measured against the appropriate MRT stations

    Next, the segments in the HDB dataset needs to be appended with the
    appropriate unique address

    After all the segments have been appended, we join all of them together
    and return as either CSV or dataframe.
    """

    logging.info("Attempting to retrieve all HDB records...")
    hdb_df = hdb.retrieve_records(data_gov_api, urls, limit)
    # Manual transformation of certain addresses due to search logic
    hdb_df['street_name'] = hdb_df['street_name'].where(hdb_df['street_name']!='MARINE CRES','MARINE CRESCENT VILLE') 

    unique_address = hdb_df['address'].unique()

    logging.info("Done retrieving all HDB records and generating unique address")
    # This takes around 13 mins
    geo_details = mrt_onemap.return_geo_many_address(onemap_api , unique_address)
    unique_address_geo = []

    #TODO: use tqdm to give an estimate on the amount of time or entries needed till completion

    logging.info("Attempting to retrieve unique addresses geo data from onemap sg ...")
    for idx, address in enumerate(unique_address):
        unique_address_geo.append([address, geo_details[idx][0],geo_details[idx][1],geo_details[idx][2]])
    unique_address_geo_pd = pd.DataFrame(unique_address_geo)
    unique_address_geo_pd.columns = ('address','full_address','lat','long')

    logging.info("Done retrieving unique addresses geo data")

    unique_address_geo_pd_valid = unique_address_geo_pd[unique_address_geo_pd['full_address'] != 'na']
    unique_address_geo_pd_valid['lat'] = unique_address_geo_pd_valid['lat'].astype('float')
    unique_address_geo_pd_valid['long'] = unique_address_geo_pd_valid['long'].astype('float')

    logging.info("Gathering non valid addresses ...")
    # This takes around 13 mins
    non_valid_addresses = [address for address in hdb_df.address.values if address in unique_address_geo_pd[unique_address_geo_pd['full_address'] == 'na'].values]
    

    logging.info("Gathering slices of HDB data ...")
    pre2000_hdb = hdb.retrieve_record(data_gov_api, resource_pre1999, limit, add_remaining_lease=True)
    yr2000to2012_hdb = hdb.retrieve_record(data_gov_api, resource_jan2000tofeb2012,limit, add_remaining_lease=True)
    yr2012to2014_hdb = hdb.retrieve_record(data_gov_api, resource_mar2012todec2014,limit, add_remaining_lease=True)
    yr2015to2016_hdb = hdb.retrieve_record(data_gov_api, resource_jan2015todec2016,limit)
    post2017_hdb = hdb.retrieve_record(data_gov_api, resource_post2017,limit)

    logging.debug(f"Sample of data from pre2000 hdb {pre2000_hdb.sample(5)}")
    logging.debug(f"Sample of data from post2017 HDB {post2017_hdb.sample(5)}")

    logging.info("Gathering MRT data ...")
    stations = mrt_onemap.extract_mrt_excelsheet(mrt_api,mrt_excel_file)
    stations_pd = mrt_onemap.return_mrt_with_geo(onemap_api ,stations)

    unique_address_geo_pd_valid = unique_address_geo_pd[unique_address_geo_pd['full_address'] != 'na']
    unique_address_geo_pd_valid['lat'] = unique_address_geo_pd_valid['lat'].astype('float')
    unique_address_geo_pd_valid['long'] = unique_address_geo_pd_valid['long'].astype('float')
    unique_address_geo_pd_valid.reset_index(drop=True)


    logging.info("Processing slices of HDB data ...")
    pre_2000_mrt_lines = ['ns', 'ew'] # or the OG green and red lines
    pattern = '|'.join(pre_2000_mrt_lines)
    logging.debug(f"Sample of stations_pd {stations_pd.head(5)}")
    stations_pre_2000 =  stations_pd[stations_pd['stn_code'].str.contains(pattern)]
    unique_address_geo_pd_valid_mrt_pre_2000 = mrt_onemap.return_closest_mrt_distances(unique_address_geo_pd_valid, stations_pre_2000)
    unique_address_geo_pd_valid_mrt = mrt_onemap.return_closest_mrt_distances(unique_address_geo_pd_valid, stations_pd)
    unique_address_geo_pd_valid_mrt.columns = ('address', 'full_address','lat','long','nearest_mrt','nearest_distance_to_mrt')
    unique_address_geo_pd_valid_mrt_pre_2000.columns = ('address', 'full_address','lat','long','nearest_mrt','nearest_distance_to_mrt')

    pre2000_hdb_valid = pre2000_hdb[~pre2000_hdb['address'].isin(non_valid_addresses)]
    pre2000_hdb_valid_w_geo = pre2000_hdb_valid.merge(unique_address_geo_pd_valid_mrt_pre_2000, left_on='address', right_on='address')

    post2000_hdb=pd.concat([yr2000to2012_hdb,yr2012to2014_hdb,yr2015to2016_hdb,post2017_hdb], axis=0)
    post2000_hdb_valid = post2000_hdb[~post2000_hdb['address'].isin(non_valid_addresses)]
    post2000_hdb_valid_w_geo = post2000_hdb_valid.merge(unique_address_geo_pd_valid_mrt, left_on='address', right_on='address')

    hdb_df_geo = pd.concat([post2000_hdb_valid_w_geo, pre2000_hdb_valid_w_geo], axis=0)
    hdb_df_geo.drop(columns='_id', inplace=True)
    logging.debug(f"Sample of data from combined HDB data with Geo information {hdb_df_geo.sample(5)}")


    ###########################################
    #######        Return Data          #######
    ###########################################
    logging.info("Writing hdb data to file ...")
    hdb_df_geo.to_csv(write_file_name)

    logging.info("Done!")


if __name__ == "__main__":
    main()