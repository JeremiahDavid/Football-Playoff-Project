
from bs4 import BeautifulSoup
from requests_html import HTMLSession # use this to get through to the embedded javascript 
import re
import numpy as np
import pandas as pd
# use this url (url_tt) to get the time to throw

def parser(url):
    
    # use HTMLSession to read the javascript embedded tables on the website
    session = HTMLSession()
    resp = session.get(url)
    
    # from here start parsing through the html using beautifulsoup
    soup = BeautifulSoup(resp.html.html, 'lxml') # (file, parsing method)
    rows = soup.find_all('tr')
    return rows


def get_team_data(rows,team_list,stat_list):
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
        for i, v in enumerate(items):
            if i == 1 or i == 4: # the indexes of the coloumns to pull data from
                if re.search('[a-zA-Z]', v.attrs['data-sort']) != None: # sees if its the name
                    team_list.append(v.attrs['data-sort'])
                else: # if its not the name its the value
                    stat_list.append(float(v.attrs['data-sort']))
            else:
                continue
    team_list = np.array(team_list)
    stat_list = np.array(stat_list)
    out = np.array([team_list, stat_list], dtype = object).T
    # out = np.concatenate((player_list.T,stat_list.T), axis = 1)
    return out
  
# the first value in each element is for the url
# the second value in each element is for the naming of the file  
stat_names = [['completions-per-game','pcpg']]
# =============================================================================
# stat_names = [['points-per-game','ppg'],['opponent-points-per-game','ppga']
#               ,['qb-sacked-per-game','qbspg'],
#               ,['4th-quarter-time-of-possession-share-pct','top4q']
#               ,['third-down-conversion-pct','trdcp']
#               ,['opponent-yards-per-play','yppa'],['giveaways-per-game','topg']
#               ,['passing-yards-per-game','pypg'],['opponent-yards-per-game','typga']
#               ,['opponent-red-zone-scores-per-game','rzspga']
#               ,['opponent-rushing-attempts-per-game','rapga']
#               ,['yards-per-game','typg'],['red-zone-scores-per-game','rzspg']
#               ,['red-zone-scoring-attempts-per-game','rzsapg']
#               ,['third-down-conversions-per-game','trdcpg']
#               ,['third-downs-per-game','trdpg']
#               ,['opponent-completions-per-game','pcpga'],
#               ,['opponent-pass-attempts-per-game','papga']
#               ,['opponent-red-zone-scoring-attempts-per-game','rzapga']
#               ,['sacks-per-game','spg'],['interceptions-thrown-per-game','itpg']
#               ,['opponent-rushing-yards-per-game','rypga']
#               ,['rushing-yards-per-game','rypg']
#               ,['interceptions-per-game','irpg'],['completions-per-game','pcpg']
#               ,['takeaways-per-game','tapg']]
# =============================================================================

years = [20,19,18,17,16,15,14,13,12,11] # DONT go to 2010 because it will mess things up 
weeks = ['09-10','09-17','09-24','10-01','10-08','10-15','10-22','10-29','11-05'
             ,'11-12','11-19','11-26','12-03','12-10','12-17','12-24','12-31']
for year in years:
    if year in [11,12,16,17,18]: # the season starts a week later (there is one game the week before that we will ignore)
        weeks = ['09-17','09-24','10-01','10-08','10-15','10-22','10-29','11-05'
             ,'11-12','11-19','11-26','12-03','12-10','12-17','12-24','12-31','01-07']
    for index,stat in enumerate(stat_names):
        # GETS THE POINTS PER GAME FOR A SPECIFIED YEAR
        
        stat_lst = []
        last_year = year - 1
            # Week 1
        url = 'https://www.teamrankings.com/nfl/stat/%s?date=20%s-%s'%(stat[0],last_year,weeks[0])
        rows = parser(url)
        stat_1 = []  # creates a list shell for the stats
        tm_1 = [] # creates a list shell for the team
        stat_1 = get_team_data(rows,tm_1,stat_1)
        stat_1 = stat_1[np.argsort(stat_1[:,0])]
        stat_lst.append(stat_1)
        
            # Week 2
        url = 'https://www.teamrankings.com/nfl/stat/%s?date=20%s-%s'%(stat[0],last_year,weeks[1])
        rows = parser(url)
        stat_2 = []  # creates a list shell for the stats
        tm_2 = [] # creates a list shell for the team
        stat_2 = get_team_data(rows,tm_2,stat_2)
        stat_2 = stat_2[np.argsort(stat_2[:,0])]
        stat_lst.append(stat_2)
        
            # Week 3
        url = 'https://www.teamrankings.com/nfl/stat/%s?date=20%s-%s'%(stat[0],last_year,weeks[2])
        rows = parser(url)
        stat_3 = []  # creates a list shell for the stats
        tm_3 = [] # creates a list shell for the team
        stat_3 = get_team_data(rows,tm_3,stat_3)
        stat_3 = stat_3[np.argsort(stat_3[:,0])]
        stat_lst.append(stat_3)
        
            # Week 4
        url = 'https://www.teamrankings.com/nfl/stat/%s?date=20%s-%s'%(stat[0],last_year,weeks[3])
        rows = parser(url)
        stat_4 = []  # creates a list shell for the stats
        tm_4 = [] # creates a list shell for the team
        stat_4 = get_team_data(rows,tm_4,stat_4)
        stat_4 = stat_4[np.argsort(stat_4[:,0])]
        stat_lst.append(stat_4)
        
            # Week 5
        url = 'https://www.teamrankings.com/nfl/stat/%s?date=20%s-%s'%(stat[0],last_year,weeks[4])
        rows = parser(url)
        stat_5 = []  # creates a list shell for the stats
        tm_5 = [] # creates a list shell for the team
        stat_5 = get_team_data(rows,tm_5,stat_5)
        stat_5 = stat_5[np.argsort(stat_5[:,0])]
        stat_lst.append(stat_5)
        
            # Week 6
        url = 'https://www.teamrankings.com/nfl/stat/%s?date=20%s-%s'%(stat[0],last_year,weeks[5])
        rows = parser(url)
        stat_6 = []  # creates a list shell for the stats
        tm_6 = [] # creates a list shell for the team
        stat_6 = get_team_data(rows,tm_6,stat_6)
        stat_6 = stat_6[np.argsort(stat_6[:,0])]
        stat_lst.append(stat_6)
        
            # Week 7
        url = 'https://www.teamrankings.com/nfl/stat/%s?date=20%s-%s'%(stat[0],last_year,weeks[6])
        rows = parser(url)
        stat_7 = []  # creates a list shell for the stats
        tm_7 = [] # creates a list shell for the team
        stat_7 = get_team_data(rows,tm_7,stat_7)
        stat_7 = stat_7[np.argsort(stat_7[:,0])]
        stat_lst.append(stat_7)
        
            # Week 8
        url = 'https://www.teamrankings.com/nfl/stat/%s?date=20%s-%s'%(stat[0],last_year,weeks[7])
        rows = parser(url)
        stat_8 = []  # creates a list shell for the stats
        tm_8 = [] # creates a list shell for the team
        stat_8 = get_team_data(rows,tm_8,stat_8)
        stat_8 = stat_8[np.argsort(stat_8[:,0])]
        stat_lst.append(stat_8)
        
            # Week 9
        url = 'https://www.teamrankings.com/nfl/stat/%s?date=20%s-%s'%(stat[0],last_year,weeks[8])
        rows = parser(url)
        stat_9 = []  # creates a list shell for the stats
        tm_9 = [] # creates a list shell for the team
        stat_9 = get_team_data(rows,tm_9,stat_9)
        stat_9 = stat_9[np.argsort(stat_9[:,0])]
        stat_lst.append(stat_9)
        
            # Week 10
        url = 'https://www.teamrankings.com/nfl/stat/%s?date=20%s-%s'%(stat[0],last_year,weeks[9])
        rows = parser(url)
        stat_10 = []  # creates a list shell for the stats
        tm_10 = [] # creates a list shell for the team
        stat_10 = get_team_data(rows,tm_10,stat_10)
        stat_10 = stat_10[np.argsort(stat_10[:,0])]
        stat_lst.append(stat_10)
        
            # Week 11
        url = 'https://www.teamrankings.com/nfl/stat/%s?date=20%s-%s'%(stat[0],last_year,weeks[10])
        rows = parser(url)
        stat_11 = []  # creates a list shell for the stats
        tm_11 = [] # creates a list shell for the team
        stat_11 = get_team_data(rows,tm_11,stat_11)
        stat_11 = stat_11[np.argsort(stat_11[:,0])]
        stat_lst.append(stat_11)
        
            # Week 12
        url = 'https://www.teamrankings.com/nfl/stat/%s?date=20%s-%s'%(stat[0],last_year,weeks[11])
        rows = parser(url)
        stat_12 = []  # creates a list shell for the stats
        tm_12 = [] # creates a list shell for the team
        stat_12 = get_team_data(rows,tm_12,stat_12)
        stat_12 = stat_12[np.argsort(stat_12[:,0])]
        stat_lst.append(stat_12)
        
            # Week 13
        url = 'https://www.teamrankings.com/nfl/stat/%s?date=20%s-%s'%(stat[0],last_year,weeks[12])
        rows = parser(url)
        stat_13 = []  # creates a list shell for the stats
        tm_13 = [] # creates a list shell for the team
        stat_13 = get_team_data(rows,tm_13,stat_13)
        stat_13 = stat_13[np.argsort(stat_13[:,0])]
        stat_lst.append(stat_13)
        
            # Week 14
        url = 'https://www.teamrankings.com/nfl/stat/%s?date=20%s-%s'%(stat[0],last_year,weeks[13])
        rows = parser(url)
        stat_14 = []  # creates a list shell for the stats
        tm_14 = [] # creates a list shell for the team
        stat_14 = get_team_data(rows,tm_14,stat_14)
        stat_14 = stat_14[np.argsort(stat_14[:,0])]
        stat_lst.append(stat_14)
        
            # Week 15
        url = 'https://www.teamrankings.com/nfl/stat/%s?date=20%s-%s'%(stat[0],last_year,weeks[14])
        rows = parser(url)
        stat_15 = []  # creates a list shell for the stats
        tm_15 = [] # creates a list shell for the team
        stat_15 = get_team_data(rows,tm_15,stat_15)
        stat_15 = stat_15[np.argsort(stat_15[:,0])]
        stat_lst.append(stat_15)
        
            # Week 16
        url = 'https://www.teamrankings.com/nfl/stat/%s?date=20%s-%s'%(stat[0],last_year,weeks[15])
        rows = parser(url)
        stat_16 = []  # creates a list shell for the stats
        tm_16 = [] # creates a list shell for the team
        stat_16 = get_team_data(rows,tm_16,stat_16)
        stat_16 = stat_16[np.argsort(stat_16[:,0])]
        stat_lst.append(stat_16)
        
            # Week 17
        url = 'https://www.teamrankings.com/nfl/stat/%s?date=20%s-%s'%(stat[0],last_year,weeks[16])
        rows = parser(url)
        stat_17 = []  # creates a list shell for the stats
        tm_17 = [] # creates a list shell for the team
        stat_17 = get_team_data(rows,tm_17,stat_17)
        stat_17 = stat_17[np.argsort(stat_17[:,0])]
        stat_lst.append(stat_17)
        
        #%%
        delcols = [] # instantiate the list of columns to delete
        stat_data = stat_lst[0] # sets the first year as the foundation 
        i = 1 # start looping after the first year
        
        while i < len(stat_lst):
            stat_data = np.hstack((stat_data,stat_lst[i]))
            delcol = 2*i
            delcols.append(delcol)
            i += 1
        # delets the repeated team name columns
        stat_data = np.delete(stat_data,delcols,1)
        stat_data = np.delete(stat_data,[0],1)
        columns = ['Week 1','Week 2','Week 3','Week 4','Week 5','Week 6','Week 7'\
                   ,'Week 8','Week 9','Week 10','Week 11','Week 12','Week 13','Week 14'\
                   ,'Week 15','Week 16','Week 17']
        index = list(stat_17[:,0])
        # converts the array to a pandas data frame
        stat_data = pd.DataFrame(stat_data, index = index, columns = columns)
        filename = '%s_20%s.csv' %(stat[1],year)
        #%%
        stat_data.to_csv(filename)








