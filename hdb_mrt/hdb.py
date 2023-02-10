""" This is a helper file for HDB data from data.gov.sg 
"""
from typing import List
import pandas as pd
import zipfile
from io import BytesIO
import requests
import logging 

def retrieve_record(url_base:str, resource_id:str, limit:int,add_remaining_lease:bool= False)-> pd.DataFrame:
    """returns a Dataframe from the base url under the resource id
    up to the limit specified. 

    Parameters
    ----------
    url_base : str
        API endpoint
    
    resource_id : 
    
    returns: pd.DataFrame
        Records of the results
    """
    payload = {
        'resource_id' : resource_id,
        'limit' : limit 
    }
    try: 
        response = requests.get(url_base, params=payload)
        data = response.json()
        data = data['result']['records'] # data only found in records under result

        data_pd = pd.DataFrame(data)
        data_pd['address'] = data_pd['block'] + ' ' + data_pd['street_name']

        if add_remaining_lease:
            data_pd['remaining_lease'] = 99 - (pd.to_datetime(data_pd['month']).dt.year - data_pd['lease_commence_date'].astype(int))

        return data_pd

    except Exception as e:
        logging.error(f'Unable to retrieve records due to {e}')

        return None

def retrieve_records(url_base:str, urls: List[str],limit:int)->pd.DataFrame:
    """Returns the entire list of url as a pandas dataframe.
    If there is a link that is broken, it will return None
    Will also add in an additional column of remaining lease into the 
    years that do not have it.

    Args:
        url_base (str): API endpoint for data.gov.sg
        urls (List[str]): list of url
        limit (int): Limits the amount of records being retrieved 
        in each given year

    Returns:
        pd.DataFrame: Compilation of the entire record
    """
    list_of_results = []
    for url in urls:
        result = retrieve_record(url_base, url, limit)
        try:
            list_of_results.append(result)

        except Exception as e:

            logging.error(f'Unable to retrieve records due to {e}')
            return None


    # 2014 and before 
    df_part1 = pd.concat((
        pd.DataFrame(list_of_results[0]),
        pd.DataFrame(list_of_results[1]),
        pd.DataFrame(list_of_results[2])), axis=0)

    # 2017 to 2014 records with additional column 'remaining lease'
    df_part2 = pd.concat((
        pd.DataFrame(list_of_results[3]),
        pd.DataFrame(list_of_results[4])), axis=0)

    df_part1['remaining_lease'] = 99 - (pd.to_datetime(df_part1['month']).dt.year - df_part1['lease_commence_date'].astype(int) )

    # Rearranging cols so that the concatation will work later
    df = df_part1[['remaining_lease']+ [col for col in df_part1.columns if col != 'remaining_lease']]
    df2 = df_part2[['remaining_lease']+ [col for col in df_part2.columns if col != 'remaining_lease']]

    df_combined = pd.concat((df, df2))

    return df_combined