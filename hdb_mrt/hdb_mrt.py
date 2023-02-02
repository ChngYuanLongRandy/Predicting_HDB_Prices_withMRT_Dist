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

import hdb
import mrt_onemap

def write_file(path:str, data:pd.DataFrame):
    data.to_csv(path)


@hydra.main(config_path="conf/", config_name="config.yaml")
def main(args):
    """
    Main function to generate the data in excel file
    """
    

    ###########################################
    ####### Initialisation of Variables #######
    ###########################################

    limit = args['specs']['limit']
    data_gov_api = args['specs']['data_gov_api']
    write_file_name = args['specs']['write_file_name']
    onemap_api = args['specs']['onemap_api']
    mrt_api = args['specs']['mrt_api']
    mrt_excel_file = args['specs']['mrt_excel_file']

    resource_post2017 = args['resource_mapping']['Jan2017-Beyond']
    resource_jan2015todec2016 = args['resource_mapping']['Jan2017-Beyond']
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

    hdb_df = hdb.retrieve_records(data_gov_api, urls, limit)
    unique_address = hdb_df['address'].unique()





    ###########################################
    #######        Append Data          #######
    ###########################################






    ###########################################
    #######        Return Data          #######
    ###########################################



if __name__ == "__main__":
    main()