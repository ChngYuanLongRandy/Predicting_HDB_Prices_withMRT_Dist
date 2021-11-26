#! python3
#Web_scrape_99.co.py scrapes 99.co website for predicting house prices
import pandas as pd
import requests, bs4, csv , re, datetime

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

#loop through each link and take all information available and store in dictionary

#Intialisation of all of the lists

NameOfProperty = ['Name_of_Property']
Web_link = ['Web Link']
Price = ['Price']
PricePerSqft =['Price_per_Sqft']
FloorLevel = ['Floor_Level']
NumberOfBedrooms = ['Number_of_Bedrooms']
Furnishing = ['Furnishing']
Facing =['Facing']
OverlookingView = ['Overlooking_View']
BuiltYear = ['Built_Year']
Tenure = ['Tenure']
PropertyType =['Property_Type']
Amenities = ['Amenities']
Description = ['Description']
DevelopmentName = ['Development_Name']
UnitTypes =['Unit_Types']
DevelopmentBuiltYear = ['Development Built Year']
DevelopmentTenure = ['Development Tenure']
Developer = ['Developer']
DevelopmentNeighbourhood = ['Development Neighbourhood']
Size = ['Size']
Toilets = ['Number of Toilets']
Misc_details = ['Misc Details']

# HDB site : https://www.99.co/singapore/sale/hdb
# Condo site : https://www.99.co/singapore/sale/condos-apartments
# Landed site : https://www.99.co/singapore/sale/houses
# EC site : https://www.99.co/singapore/sale/executive-condominiums

website_main_link = 'http://99.co'

website_links = ['https://www.99.co/singapore/sale/hdb',
                 'https://www.99.co/singapore/sale/condos-apartments',
                 'https://www.99.co/singapore/sale/houses',
                 'https://www.99.co/singapore/sale/executive-condominiums']

# no search terms

link = website_links[0]

#while next_url != 'https://www.jobstreet.com.sg/en/job-search/data-scientist-jobs/1/':

#for link in website_links:
print('Searching in \n', link , '\n now')

#individual a href buried in div class _2kH6B for HDB, Condo, Landed and EC
res = requests.get(link)
res.raise_for_status()
soup = bs4.BeautifulSoup(res.text, 'html.parser')

# this should pick out each job in the page(listing), each page will have a maximum of 34 jobs
results = soup.find_all('div', class_='_2kH6B')

print('total results ', len(results))

for index, listing in enumerate(results):
    print('--' * 30)
    print('Entering Main Loop in Listing, index number ', index)
    print('--' * 30,'\n')
    if index > 5:
        break
    else:
        #there should only be one link in the div class above
        result_link = listing.find('a').get('href')
        #result_link = listing.select_one('a').get('href')
        result_text = listing.find('a').text
        print('Web link is ', website_main_link+result_link) # for getting link
        print('Name of Property is ', result_text) # for getting text between the link
        # Name can be found on listing or in the result link
        NameOfProperty.append(result_text)
        Web_link.append(website_main_link + result_link)

    # All of the links would have been collected in Web_link
    # All of the names would have been collected in NameOfProperty

# Going into each individual link
for index, result_link in enumerate(Web_link):
    print('Entering Sub loop in job page')
    # the first value of web link is the header
    if index == 0:
        continue
    else:
        print('*'*30)
        print('Going into individual link number ', index)
        print('Going into individual link ', result_link)
        print('*'*30 ,'\n')
        listing_result_link = requests.get(result_link)
        try:
            listing_result_link.raise_for_status()
        except Exception as exc:
            print('Problem due to exception ' + str(exc))
        listing_result_data = bs4.BeautifulSoup(listing_result_link.text, 'html.parser')
        #listing_result_data_price = listing_result_data.select_one('div', id_='price').getText()
        #print(listing_result_data.select_one('div', id_='price').getText())
        listing_result_data_price = listing_result_data.select_one('h2', class_='_1zGm8 _3na6W _1vzK2').getText()
        print('Price is ', listing_result_data_price)
        listing_result_data_name = listing_result_data.select_one('h1', class_='_3Wogd JMF8h lFqTi _1vzK2').getText()
        #Getting all text data in tag 'p', class_='_2sIc2 _29qfj _2rhE-'.
        #This includes the number of bedrooms, bathrooms, area space etc
        listing_result_data_Misc = listing_result_data.find_all('p', class_="_2sIc2 _29qfj _2rhE-")
        #temp_string = ''
        temp_list = [p.getText() for p in listing_result_data_Misc]
            # print(p.getText())
            # temp_string = ' , '.join(p.getText())
        print(temp_list)
        Misc_details.append(temp_list)
        print('Price is ',listing_result_data_price)
        print('Name of property is ', listing_result_data_name)
        Price.append(listing_result_data_price)

def inspect_values_inlist(a_list, number):
    print('Print first ', number ,' Results of ', a_list[0])
    print('Length of ',a_list[0], ' is ', len(a_list))
    for i in range(1,number+1):
        print(a_list[i])
    print('\n')

inspect_values_inlist(NameOfProperty, 5)
inspect_values_inlist(Price, 5)
inspect_values_inlist(Web_link, 5)
inspect_values_inlist(Misc_details,5)


# Expected output should be 34. There are promoted listings that are not captured which is good

#print('Number of results is ', str(len(result)))
#print('Printing searches of result ', result[0])