
from bs4 import BeautifulSoup
from requests_html import HTMLSession # use this to get through to the embedded javascript 
import numpy as np
import pandas as pd
import re
# use this url (url_tt) to get the time to throw

# This gets the names of the teams in the correct format
def switcher(string):
    splitter = string.rsplit(' ',1) # split('what to split on','max splits to make')
    if splitter[1] == 'Rams' or splitter[1] == 'Chargers':
        string = 'LA %s' %(splitter[1])
    elif splitter[1] == 'Jets' or splitter[1] == 'Giants':
        string = 'NY %s' %(splitter[1])
    else:
        string = splitter[0]
    return string


def parser(url):
    
    # use HTMLSession to read the javascript embedded tables on the website
    session = HTMLSession()
    resp = session.get(url)
    
    # from here start parsing through the html using beautifulsoup
    soup = BeautifulSoup(resp.html.html, 'lxml') # (file, parsing method)
    rows = soup.find_all('tr')
    return rows


def get_team_data(rows,team_list):
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
        items = row.find_all('a') # The elements in the row
        # print(row.prettify())
        for i, v in enumerate(items):
            # test = v.contents[0]
            # print(v.contents) # this HTML is different than the other parts of the website. Use .contents to get the contents inside the "a" level. 
            if v.contents[0] != 'boxscore': # the index of the second coloumn to pull data from
                team = switcher(v.contents[0])
                team_list.append(team)
            else:
                continue
    team_list = np.array([team_list], dtype = object).T
    # out = np.array([team_list], dtype = object).T
    return team_list
    
years = [20,19,18,17,16,15,14,13,12,11]
for year in years:
    last_year = year - 1
    
    # GETS THE OPPONENTS PLAYED FOR THE YEAR
    team_lst = []
    
        # Arizona
    url = 'https://www.pro-football-reference.com/teams/crd/20%s.htm'%(last_year)
    rows = parser(url)
    team_1 = []  # creates a list shell for the teams
    team_1 = get_team_data(rows,team_1)
    team_lst.append(team_1)
    
        # Atlanta
    url = 'https://www.pro-football-reference.com/teams/atl/20%s.htm'%(last_year)
    rows = parser(url)
    team_2 = []  # creates a list shell for the teams
    team_2 = get_team_data(rows,team_2)
    team_lst.append(team_2)
    
        # Baltimore
    url = 'https://www.pro-football-reference.com/teams/rav/20%s.htm'%(last_year)
    rows = parser(url)
    team_3 = []  # creates a list shell for the teams
    team_3 = get_team_data(rows,team_3)
    team_lst.append(team_3)
    
        # Buffalo
    url = 'https://www.pro-football-reference.com/teams/buf/20%s.htm'%(last_year)
    rows = parser(url)
    team_4 = []  # creates a list shell for the teams
    team_4 = get_team_data(rows,team_4)
    team_lst.append(team_4)
    
        # Carolina
    url = 'https://www.pro-football-reference.com/teams/car/20%s.htm'%(last_year)
    rows = parser(url)
    team_5 = []  # creates a list shell for the teams
    team_5 = get_team_data(rows,team_5)
    team_lst.append(team_5)
    
        # Chicago
    url = 'https://www.pro-football-reference.com/teams/chi/20%s.htm'%(last_year)
    rows = parser(url)
    team_6 = []  # creates a list shell for the teams
    team_6 = get_team_data(rows,team_6)
    team_lst.append(team_6)
    
        # Cincinnati
    url = 'https://www.pro-football-reference.com/teams/cin/20%s.htm'%(last_year)
    rows = parser(url)
    team_7 = []  # creates a list shell for the teams
    team_7 = get_team_data(rows,team_7)
    team_lst.append(team_7)
    
        # Cleveland
    url = 'https://www.pro-football-reference.com/teams/cle/20%s.htm'%(last_year)
    rows = parser(url)
    team_8 = []  # creates a list shell for the teams
    team_8 = get_team_data(rows,team_8)
    team_lst.append(team_8)
    
        # Dallas
    url = 'https://www.pro-football-reference.com/teams/dal/20%s.htm'%(last_year)
    rows = parser(url)
    team_9 = []  # creates a list shell for the teams
    team_9 = get_team_data(rows,team_9)
    team_lst.append(team_9)
    
        # Denver
    url = 'https://www.pro-football-reference.com/teams/den/20%s.htm'%(last_year)
    rows = parser(url)
    team_10 = []  # creates a list shell for the teams
    team_10 = get_team_data(rows,team_10)
    team_lst.append(team_10)
    
        # Detroit
    url = 'https://www.pro-football-reference.com/teams/det/20%s.htm'%(last_year)
    rows = parser(url)
    team_11 = []  # creates a list shell for the teams
    team_11 = get_team_data(rows,team_11)
    team_lst.append(team_11)
    
        # Green Bay
    url = 'https://www.pro-football-reference.com/teams/gnb/20%s.htm'%(last_year)
    rows = parser(url)
    team_12 = []  # creates a list shell for the teams
    team_12 = get_team_data(rows,team_12)
    team_lst.append(team_12)
    
        # Houston
    url = 'https://www.pro-football-reference.com/teams/htx/20%s.htm'%(last_year)
    rows = parser(url)
    team_13 = []  # creates a list shell for the teams
    team_13 = get_team_data(rows,team_13)
    team_lst.append(team_13)
    
        # Indianapolis
    url = 'https://www.pro-football-reference.com/teams/clt/20%s.htm'%(last_year)
    rows = parser(url)
    team_14 = []  # creates a list shell for the teams
    team_14 = get_team_data(rows,team_14)
    team_lst.append(team_14)
    
        # Jacksonville
    url = 'https://www.pro-football-reference.com/teams/jax/20%s.htm'%(last_year)
    rows = parser(url)
    team_15 = []  # creates a list shell for the teams
    team_15 = get_team_data(rows,team_15)
    team_lst.append(team_15)
    
        # Kansas City
    url = 'https://www.pro-football-reference.com/teams/kan/20%s.htm'%(last_year)
    rows = parser(url)
    team_16 = []  # creates a list shell for the teams
    team_16 = get_team_data(rows,team_16)
    team_lst.append(team_16)
    
        # LA Chargers
    url = 'https://www.pro-football-reference.com/teams/sdg/20%s.htm'%(last_year)
    rows = parser(url)
    team_17 = []  # creates a list shell for the teams
    team_17 = get_team_data(rows,team_17)
    team_lst.append(team_17)
    
        # LA Rams
    url = 'https://www.pro-football-reference.com/teams/ram/20%s.htm'%(last_year)
    rows = parser(url)
    team_18 = []  # creates a list shell for the teams
    team_18 = get_team_data(rows,team_18)
    team_lst.append(team_18)
    
        # Miami
    url = 'https://www.pro-football-reference.com/teams/mia/20%s.htm'%(last_year)
    rows = parser(url)
    team_19 = []  # creates a list shell for the teams
    team_19 = get_team_data(rows,team_19)
    team_lst.append(team_19)
    
        # Minnesota
    url = 'https://www.pro-football-reference.com/teams/min/20%s.htm'%(last_year)
    rows = parser(url)
    team_20 = []  # creates a list shell for the teams
    team_20 = get_team_data(rows,team_20)
    team_lst.append(team_20)
    
        # NY Giants
    url = 'https://www.pro-football-reference.com/teams/nyg/20%s.htm'%(last_year)
    rows = parser(url)
    team_21 = []  # creates a list shell for the teams
    team_21 = get_team_data(rows,team_21)
    team_lst.append(team_21)
    
        # NY Jets
    url = 'https://www.pro-football-reference.com/teams/nyj/20%s.htm'%(last_year)
    rows = parser(url)
    team_22 = []  # creates a list shell for the teams
    team_22 = get_team_data(rows,team_22)
    team_lst.append(team_22)
    
    
        # New England
    url = 'https://www.pro-football-reference.com/teams/nwe/20%s.htm'%(last_year)
    rows = parser(url)
    team_23 = []  # creates a list shell for the teams
    team_23 = get_team_data(rows,team_23)
    team_lst.append(team_23)
    
    
        # New Orleans
    url = 'https://www.pro-football-reference.com/teams/nor/20%s.htm'%(last_year)
    rows = parser(url)
    team_24 = []  # creates a list shell for the teams
    team_24 = get_team_data(rows,team_24)
    team_lst.append(team_24)
    
        # Oakland
    url = 'https://www.pro-football-reference.com/teams/rai/20%s.htm'%(last_year)
    rows = parser(url)
    team_25 = []  # creates a list shell for the teams
    team_25 = get_team_data(rows,team_25)
    team_lst.append(team_25)
    
        # Philadelphia
    url = 'https://www.pro-football-reference.com/teams/phi/20%s.htm'%(last_year)
    rows = parser(url)
    team_26 = []  # creates a list shell for the teams
    team_26 = get_team_data(rows,team_26)
    team_lst.append(team_26)
    
        # Pittsburgh
    url = 'https://www.pro-football-reference.com/teams/pit/20%s.htm'%(last_year)
    rows = parser(url)
    team_27 = []  # creates a list shell for the teams
    team_27 = get_team_data(rows,team_27)
    team_lst.append(team_27)
    
        # San Francisco
    url = 'https://www.pro-football-reference.com/teams/sfo/20%s.htm'%(last_year)
    rows = parser(url)
    team_28 = []  # creates a list shell for the teams
    team_28 = get_team_data(rows,team_28)
    team_lst.append(team_28)
    
        # Seattle
    url = 'https://www.pro-football-reference.com/teams/sea/20%s.htm'%(last_year)
    rows = parser(url)
    team_29 = []  # creates a list shell for the teams
    team_29 = get_team_data(rows,team_29)
    team_lst.append(team_29)
    
        # Tampa Bay
    url = 'https://www.pro-football-reference.com/teams/tam/20%s.htm'%(last_year)
    rows = parser(url)
    team_30 = []  # creates a list shell for the teams
    team_30 = get_team_data(rows,team_30)
    team_lst.append(team_30)
    
        # Tennessee
    url = 'https://www.pro-football-reference.com/teams/oti/20%s.htm'%(last_year)
    rows = parser(url)
    team_31 = []  # creates a list shell for the teams
    team_31 = get_team_data(rows,team_31)
    team_lst.append(team_31)
    
        # Washington
    url = 'https://www.pro-football-reference.com/teams/was/20%s.htm'%(last_year)
    rows = parser(url)
    team_32 = []  # creates a list shell for the teams
    team_32 = get_team_data(rows,team_32)
    team_lst.append(team_32)
    
    team_data = np.empty([16,1]) # sets the first year as the foundation 
    i = 0 # start looping after the first year
    
    # condense the games to strictly the regular season
    while i < len(team_lst):
        check = team_lst[i] # pulls the numpy array to be inspected
        while len(check) > 16: # if the size is more than 16, delete the elements at the end of the array
            check = np.delete(check,-1, axis =0) # deletes the post season games
        team_data = np.hstack((team_data,check))
        i += 1
    team_data = np.delete(team_data,0,axis=1)
    columns = ['Arizona','Atlanta','Baltimore','Buffalo','Carolina','Chicago'\
                   ,'Cincinnati','Cleveland','Dallas','Denver','Detroit','Green Bay'\
                   ,'Houston','Indianapolis','Jacksonville','Kansas City','LA Chargers'\
                   ,'LA Rams','Miami','Minnesota','NY Giants','NY Jets','New England'\
                   ,'New Orleans','Oakland','Philadelphia','Pittsburgh','San Francisco'\
                   ,'Seattle','Tampa Bay','Tennessee','Washington']
    # converts the array to a pandas data frame
    team_data = pd.DataFrame(team_data, columns = columns)
    filename = 'opponents_20%s.csv' %(year)
    team_data.to_csv(filename)








