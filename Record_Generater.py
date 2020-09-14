
from bs4 import BeautifulSoup
from requests_html import HTMLSession # use this to get through to the embedded javascript 
import numpy as np
import pandas as pd
import re
# use this url (url_tt) to get the time to throw


def parser(url):
    
    # use HTMLSession to read the javascript embedded tables on the website
    session = HTMLSession()
    resp = session.get(url)
    
    # from here start parsing through the html using beautifulsoup
    soup = BeautifulSoup(resp.html.html, 'lxml') # (file, parsing method)
    rows = soup.find_all('tr')
    return rows


def get_team_data(rows,record):
    """
    This function takes in a table and loops through the rows and 
    then loops through the elements in each row to get a team and stat.
    
    
    Parameters
    ----------
    rows : bs4.element module 
        the meat. what is being scraped through.
    stat_list : list
        list of stats that eventually get turned into a column in an array
    team_list : list
        list of teams that evenually gets turned into a column in an array
        

    Returns
    -------
    out : numpy array
        an array [team, stat]

    """
    for row in rows: 
        items = row.find_all('td') # The elements in the row
        # print(row.prettify())
        for i, v in enumerate(items):
            # print(v.contents) # this HTML is different than the other parts of the website. Use .contents to get the contents inside the "td" level. 
            if v['data-stat'] == 'game_outcome' and len(v.contents) > 0: 
                if v.contents[0] == 'W':
                    record.append(1) # a 1 in the list will represent a win
                elif v.contents[0] == 'L':
                    record.append(0) # a 0 in the list will represent a loss
                elif v.contents[0] == 'T':
                    record.append(0) # a 0 in the list will represent a tie
            else:
                continue
    record = np.array([record],dtype = object).T
    return record
    
years = [20,19,18,17,16,15,14,13,12,11]
for year in years:
    last_year = year - 1
    
    # GETS THE OPPONENTS PLAYED FOR THE YEAR
    record_lst = []
    
        # Arizona
    url = 'https://www.pro-football-reference.com/teams/crd/20%s.htm'%(last_year)
    rows = parser(url)
    record_1 = []  # creates a list shell for the records
    record_1 = get_team_data(rows,record_1)
    record_lst.append(record_1)
    
        # Atlanta
    url = 'https://www.pro-football-reference.com/teams/atl/20%s.htm'%(last_year)
    rows = parser(url)
    record_2 = []  # creates a list shell for the records
    record_2 = get_team_data(rows,record_2)
    record_lst.append(record_2)
    
        # Baltimore
    url = 'https://www.pro-football-reference.com/teams/rav/20%s.htm'%(last_year)
    rows = parser(url)
    record_3 = []  # creates a list shell for the records
    record_3 = get_team_data(rows,record_3)
    record_lst.append(record_3)
    
        # Buffalo
    url = 'https://www.pro-football-reference.com/teams/buf/20%s.htm'%(last_year)
    rows = parser(url)
    record_4 = []  # creates a list shell for the records
    record_4 = get_team_data(rows,record_4)
    record_lst.append(record_4)
    
        # Carolina
    url = 'https://www.pro-football-reference.com/teams/car/20%s.htm'%(last_year)
    rows = parser(url)
    record_5 = []  # creates a list shell for the records
    record_5 = get_team_data(rows,record_5)
    record_lst.append(record_5)
    
        # Chicago
    url = 'https://www.pro-football-reference.com/teams/chi/20%s.htm'%(last_year)
    rows = parser(url)
    record_6 = []  # creates a list shell for the records
    record_6 = get_team_data(rows,record_6)
    record_lst.append(record_6)
    
        # Cincinnati
    url = 'https://www.pro-football-reference.com/teams/cin/20%s.htm'%(last_year)
    rows = parser(url)
    record_7 = []  # creates a list shell for the records
    record_7 = get_team_data(rows,record_7)
    record_lst.append(record_7)
    
        # Cleveland
    url = 'https://www.pro-football-reference.com/teams/cle/20%s.htm'%(last_year)
    rows = parser(url)
    record_8 = []  # creates a list shell for the records
    record_8 = get_team_data(rows,record_8)
    record_lst.append(record_8)
    
        # Dallas
    url = 'https://www.pro-football-reference.com/teams/dal/20%s.htm'%(last_year)
    rows = parser(url)
    record_9 = []  # creates a list shell for the records
    record_9 = get_team_data(rows,record_9)
    record_lst.append(record_9)
    
        # Denver
    url = 'https://www.pro-football-reference.com/teams/den/20%s.htm'%(last_year)
    rows = parser(url)
    record_10 = []  # creates a list shell for the records
    record_10 = get_team_data(rows,record_10)
    record_lst.append(record_10)
    
        # Detroit
    url = 'https://www.pro-football-reference.com/teams/det/20%s.htm'%(last_year)
    rows = parser(url)
    record_11 = []  # creates a list shell for the records
    record_11 = get_team_data(rows,record_11)
    record_lst.append(record_11)
    
        # Green Bay
    url = 'https://www.pro-football-reference.com/teams/gnb/20%s.htm'%(last_year)
    rows = parser(url)
    record_12 = []  # creates a list shell for the records
    record_12 = get_team_data(rows,record_12)
    record_lst.append(record_12)
    
        # Houston
    url = 'https://www.pro-football-reference.com/teams/htx/20%s.htm'%(last_year)
    rows = parser(url)
    record_13 = []  # creates a list shell for the records
    record_13 = get_team_data(rows,record_13)
    record_lst.append(record_13)
    
        # Indianapolis
    url = 'https://www.pro-football-reference.com/teams/clt/20%s.htm'%(last_year)
    rows = parser(url)
    record_14 = []  # creates a list shell for the records
    record_14 = get_team_data(rows,record_14)
    record_lst.append(record_14)
    
        # Jacksonville
    url = 'https://www.pro-football-reference.com/teams/jax/20%s.htm'%(last_year)
    rows = parser(url)
    record_15 = []  # creates a list shell for the records
    record_15 = get_team_data(rows,record_15)
    record_lst.append(record_15)
    
        # Kansas City
    url = 'https://www.pro-football-reference.com/teams/kan/20%s.htm'%(last_year)
    rows = parser(url)
    record_16 = []  # creates a list shell for the records
    record_16 = get_team_data(rows,record_16)
    record_lst.append(record_16)
    
        # LA Chargers
    url = 'https://www.pro-football-reference.com/teams/sdg/20%s.htm'%(last_year)
    rows = parser(url)
    record_17 = []  # creates a list shell for the records
    record_17 = get_team_data(rows,record_17)
    record_lst.append(record_17)
    
        # LA Rams
    url = 'https://www.pro-football-reference.com/teams/ram/20%s.htm'%(last_year)
    rows = parser(url)
    record_18 = []  # creates a list shell for the records
    record_18 = get_team_data(rows,record_18)
    record_lst.append(record_18)
    
        # Miami
    url = 'https://www.pro-football-reference.com/teams/mia/20%s.htm'%(last_year)
    rows = parser(url)
    record_19 = []  # creates a list shell for the records
    record_19 = get_team_data(rows,record_19)
    record_lst.append(record_19)
    
        # Minnesota
    url = 'https://www.pro-football-reference.com/teams/min/20%s.htm'%(last_year)
    rows = parser(url)
    record_20 = []  # creates a list shell for the records
    record_20 = get_team_data(rows,record_20)
    record_lst.append(record_20)
    
        # NY Giants
    url = 'https://www.pro-football-reference.com/teams/nyg/20%s.htm'%(last_year)
    rows = parser(url)
    record_21 = []  # creates a list shell for the records
    record_21 = get_team_data(rows,record_21)
    record_lst.append(record_21)
    
        # NY Jets
    url = 'https://www.pro-football-reference.com/teams/nyj/20%s.htm'%(last_year)
    rows = parser(url)
    record_22 = []  # creates a list shell for the records
    record_22 = get_team_data(rows,record_22)
    record_lst.append(record_22)
    
    
        # New England
    url = 'https://www.pro-football-reference.com/teams/nwe/20%s.htm'%(last_year)
    rows = parser(url)
    record_23 = []  # creates a list shell for the records
    record_23 = get_team_data(rows,record_23)
    record_lst.append(record_23)
    
    
        # New Orleans
    url = 'https://www.pro-football-reference.com/teams/nor/20%s.htm'%(last_year)
    rows = parser(url)
    record_24 = []  # creates a list shell for the records
    record_24 = get_team_data(rows,record_24)
    record_lst.append(record_24)
    
        # Oakland
    url = 'https://www.pro-football-reference.com/teams/rai/20%s.htm'%(last_year)
    rows = parser(url)
    record_25 = []  # creates a list shell for the records
    record_25 = get_team_data(rows,record_25)
    record_lst.append(record_25)
    
        # Philadelphia
    url = 'https://www.pro-football-reference.com/teams/phi/20%s.htm'%(last_year)
    rows = parser(url)
    record_26 = []  # creates a list shell for the records
    record_26 = get_team_data(rows,record_26)
    record_lst.append(record_26)
    
        # Pittsburgh
    url = 'https://www.pro-football-reference.com/teams/pit/20%s.htm'%(last_year)
    rows = parser(url)
    record_27 = []  # creates a list shell for the records
    record_27 = get_team_data(rows,record_27)
    record_lst.append(record_27)
    
        # San Francisco
    url = 'https://www.pro-football-reference.com/teams/sfo/20%s.htm'%(last_year)
    rows = parser(url)
    record_28 = []  # creates a list shell for the records
    record_28 = get_team_data(rows,record_28)
    record_lst.append(record_28)
    
        # Seattle
    url = 'https://www.pro-football-reference.com/teams/sea/20%s.htm'%(last_year)
    rows = parser(url)
    record_29 = []  # creates a list shell for the records
    record_29 = get_team_data(rows,record_29)
    record_lst.append(record_29)
    
        # Tampa Bay
    url = 'https://www.pro-football-reference.com/teams/tam/20%s.htm'%(last_year)
    rows = parser(url)
    record_30 = []  # creates a list shell for the records
    record_30 = get_team_data(rows,record_30)
    record_lst.append(record_30)
    
        # Tennessee
    url = 'https://www.pro-football-reference.com/teams/oti/20%s.htm'%(last_year)
    rows = parser(url)
    record_31 = []  # creates a list shell for the records
    record_31 = get_team_data(rows,record_31)
    record_lst.append(record_31)
    
        # Washington
    url = 'https://www.pro-football-reference.com/teams/was/20%s.htm'%(last_year)
    rows = parser(url)
    record_32 = []  # creates a list shell for the records
    record_32 = get_team_data(rows,record_32)
    record_lst.append(record_32)
    
    record_data = np.empty([16,1]) # sets the first year as the foundation 
    i = 0 # start looping after the first year
    
    # condense the games to strictly the regular season
    while i < len(record_lst):
        check = record_lst[i] # pulls the numpy array to be inspected
        while len(check) > 16: # if the size is more than 16, delete the elements at the end of the array
            check = np.delete(check,-1, axis =0) # deletes the post season games
        record_data = np.hstack((record_data,check))
        i += 1
    record_data = np.delete(record_data,0,axis=1)
    columns = ['Arizona','Atlanta','Baltimore','Buffalo','Carolina','Chicago'\
                   ,'Cincinnati','Cleveland','Dallas','Denver','Detroit','Green Bay'\
                   ,'Houston','Indianapolis','Jacksonville','Kansas City','LA Chargers'\
                   ,'LA Rams','Miami','Minnesota','NY Giants','NY Jets','New England'\
                   ,'New Orleans','Oakland','Philadelphia','Pittsburgh','San Francisco'\
                   ,'Seattle','Tampa Bay','Tennessee','Washington']
    # converts the array to a pandas data frame
    record_data = pd.DataFrame(record_data, columns = columns)
    filename = 'records_20%s.csv' %(year)
    record_data.to_csv(filename)








