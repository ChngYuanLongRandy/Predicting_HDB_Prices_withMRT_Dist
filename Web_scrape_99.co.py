#! python3
# Web_scrape_99.co.py scrapes 99.co website for predicting house prices

import pandas as pd
import requests, bs4, datetime, csv, time

# 99.co Robots.txt
# User-agent: trovitBot
# Disallow: /
#
# User-agent: rogerbot
# Crawl-delay: 3
#
# User-agent: dotbot
# Crawl-delay: 3
#
# User-agent: *
# Sitemap: https://www.99.co/singapore/sitemap.xml
# Sitemap: https://www.99.co/id/sitemap.xml
# Disallow: /api
# Disallow: /id/account/
# Disallow: /id/ajax/
# Disallow: /id/ajax2/
# Disallow: /id/cart/
# Disallow: /id/assets/
# Disallow: /id/images/
# Disallow: /id/komunitas/ajax/
# Disallow: /id/ad/
# Disallow: /id/administration/
# Disallow: /id/api/
# Disallow: /id/logout
# Disallow: /id/signup
# Disallow: /id/ezy/
# Disallow: /id/tracker.gif
# Disallow: /id/property/ajax-social-share
# Disallow: /id/guide/analytics
# Disallow: /id/property/report-ad
# Disallow: /id/komunitas/forums/create-topic
# Disallow: /id/property/quick-nav-load
# Disallow: /id/search/histogram
# Disallow: /id/p/
# Disallow: /id/*/pdf
# Disallow: /singapore/rent/mrt-stations
# Disallow: /singapore/sale/mrt-stations
# Disallow: /singapore/s/rent/mrt-stations
# Disallow: /singapore/s/sale/mrt-stations
# Disallow: /singapore/rent/schools
# Disallow: /singapore/sale/schools
# Disallow: /singapore/s/rent/schools
# Disallow: /singapore/s/sale/schools
# Disallow: /singapore/rent/travel-time
# Disallow: /singapore/sale/travel-time
# Disallow: /singapore/s/rent/travel-time
# Disallow: /singapore/s/sale/travel-time
# Disallow: /singapore/new-launches/marina-one-residence

# loop through each link and take all information available and store in dictionary

# Intialisation of all of the lists

NameOfProperty = ['Name_of_Property']
Web_link = ['Web Link']
Price = ['Price']
PricePerSqft = ['Price_per_Sqft']
FloorLevel = ['Floor_Level']
NumberOfBedrooms = ['Number_of_Bedrooms']
Furnishing = ['Furnishing']
Facing = ['Facing']
OverlookingView = ['Overlooking_View']
BuiltYear = ['Built_Year']
Tenure = ['Tenure']
PropertyType = ['Property_Type']
Amenities = ['Amenities']
Description = ['Description']
DevelopmentName = ['Development_Name']
UnitTypes = ['Unit_Types']
DevelopmentBuiltYear = ['Development Built Year']
DevelopmentTenure = ['Development Tenure']
Developer = ['Developer']
DevelopmentNeighbourhood = ['Development Neighbourhood']
Size = ['Size']
Toilets = ['Number of Toilets']
Misc_details = ['Misc Details']
Misc_details2 = ['Everything in Property Details and Development Overview']
SpecialProperties = ['Closeness to MRT and Insights']

# HDB site : https://www.99.co/singapore/sale/hdb
# Condo site : https://www.99.co/singapore/sale/condos-apartments
# Landed site : https://www.99.co/singapore/sale/houses
# EC site : https://www.99.co/singapore/sale/executive-condominiums


# linkies
# should the main link have an s? https?
website_main_link = 'http://99.co'

website_links = ['https://www.99.co/singapore/sale/hdb',
                 'https://www.99.co/singapore/sale/condos-apartments',
                 'https://www.99.co/singapore/sale/houses',
                 'https://www.99.co/singapore/sale/executive-condominiums']

link = website_links[3]
first_page_bool = True
page_counter = 2


# 'https://www.99.co/singapore/sale/hdb?page_num=2'
# 'https://www.99.co/singapore/sale/condos-apartments?page_num=3'
# 'https://www.99.co/singapore/sale/executive-condominiums?page_num=2'
# 'https://www.99.co/singapore/sale/houses?page_num=2'

def write_output():
    data = zip(*(NameOfProperty,
                 Price,
                 Web_link,
                 Misc_details,
                 Misc_details2,
                 Amenities,
                 SpecialProperties))

    file = open("99co_scrape_{}.csv".format(datetime.date.today()), 'w', newline='', encoding='utf-8')
    wr = csv.writer(file, quoting=csv.QUOTE_ALL)
    for item in data:
        wr.writerow(item)


def inspect_values_inlist(a_list, number):
    print('Print first ', number, ' Results of ', a_list[0])
    print('Length of ', a_list[0], ' is ', len(a_list))
    for i in range(1, number + 1):
        print(a_list[i])
    print('\n')


def listing_loop(link):
    print('Entering listing function')
    # individual a href buried in div class _2kH6B for HDB, Condo, Landed and EC
    res = requests.get(link)
    res.raise_for_status()
    soup = bs4.BeautifulSoup(res.text, 'html.parser')

    # this should pick out each job in the page(listing), each page will have a maximum of 34 jobs
    results = soup.find_all('div', class_='_2kH6B')
    print('total results ', len(results))
    # Main Loop
    global first_page_bool, page_counter

    for index, listing in enumerate(results):
        print('\n','--' * 30)
        print('Entering Main Loop in Listing, index number ', index)
        print('--' * 30, '\n')

        # The following is for stopping the running before it gets too big
        # if index > 10:
        #     break
        # else:

        # there should only be one link in the div class above
        result_link = listing.find('a').get('href')
        # result_link = listing.select_one('a').get('href')
        result_text = listing.find('a').text
        print('Web link is ', website_main_link + result_link)  # for getting link
        print('Name of Property is ', result_text)  # for getting text between the link
        # Name can be found on listing or in the result link
        NameOfProperty.append(result_text)
        result_link = website_main_link + result_link
        Web_link.append(result_link)

        # The link would have been collected in Web_link
        # The name would have been collected in NameOfProperty

        print('\n','*' * 30)
        print('Going into individual link ', result_link)
        print('*' * 30, '\n')

        # Going into the individual link
        listing_result_link = requests.get(result_link)
        try:
            listing_result_link.raise_for_status()
        except Exception as exc:
            print('Problem due to exception ' + str(exc))

        listing_result_data = bs4.BeautifulSoup(listing_result_link.text, 'html.parser')

        # Price
        listing_result_data_price = listing_result_data.select_one('h2', class_='_1zGm8 _3na6W _1vzK2').getText()
        print('Price is ', listing_result_data_price)
        Price.append(listing_result_data_price)

        # Name? Already contained from main listing
        listing_result_data_name = listing_result_data.select_one('h1', class_='_3Wogd JMF8h lFqTi _1vzK2').getText()
        print('Name of property is ', listing_result_data_name)

        # Misc Details
        # Getting all text data in tag 'p', class_='_2sIc2 _29qfj _2rhE-'.
        # This includes the number of bedrooms, bathrooms, area space etc
        listing_result_data_Misc = listing_result_data.find_all('p', class_="_2sIc2 _29qfj _2rhE-")
        temp_list = [p.getText() for p in listing_result_data_Misc]
        print(temp_list)
        Misc_details.append(temp_list)

        # Everything in Property Details and Development Overview
        listing_result_data_Misc2 = listing_result_data.find_all('div', class_='_2dry3')
        temp_list2 = [tag.getText() for tag in listing_result_data_Misc2]
        print('Everything in Property Details and Development Overview ', temp_list2)
        Misc_details2.append(temp_list2)

        # Amentities
        listing_result_data_Amenities = listing_result_data.find_all('div', class_="_3atmT")
        temp_list3 = [p.getText() for p in listing_result_data_Amenities]
        print('Amentities ', temp_list3)
        Amenities.append(temp_list3)

        # MRT
        listing_result_data_SpecialProperties = listing_result_data.find_all('p', class_="_2sIc2 _2rhE- _1c-pJ")
        temp_list4 = [p.getText() for p in listing_result_data_SpecialProperties]
        print('MRT ', temp_list4)
        SpecialProperties.append(temp_list4)

    # End of Job Listing

# for link in website_links:


# Start of Main Body
# Checks if it is the first page? -> Go run listing loop function
# if not first page -> Checks page counter
# if page counter < specified number, fetch next link and run listing loop function

def run_main_sequence():
    global page_counter, hdb_next_link
    print('First page bool status is ', first_page_bool)
    print('Page counter is ', page_counter)
    print('Searching in \n', link, '\n now')

    listing_loop(link)
    # if this is the first page then it will set it to false
    # then once its not the first page it will increase
    # i.e First encounter sets to false, second encounter, ups the page counter

    print('First page bool status is ', first_page_bool)
    print('Page counter is ', page_counter)

    while page_counter <50 :

        hdb_next_link = website_main_link + '/singapore/sale/executive-condominiums?page_num=' + str(page_counter)
        time.sleep(10)
        print('*-' * 30)
        print('Attempting next link ', hdb_next_link)
        print('*-' * 30, '\n')
        listing_loop(hdb_next_link)
        page_counter += 1

def main():
    run_main_sequence()
    # Outputs to csv
    write_output()

main()
