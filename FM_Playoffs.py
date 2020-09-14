import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression



def NFL_Model(year,k):
    yr = str(year)
    # this data is from 'pro football reference'
    ppg = pd.read_csv('ppg_%s.csv' %(yr)).T # points per game data
    ppga = pd.read_csv('ppga_%s.csv' %(yr)).T # points per game allowed data
    pypg = pd.read_csv('pypg_%s.csv' %(yr)).T # passing yards per game
    pypga = pd.read_csv('pypga_%s.csv' %(yr)).T # passing yards per game allowed
    rypg = pd.read_csv('rypg_%s.csv' %(yr)).T # rushing yards per game
    rypga = pd.read_csv('rypga_%s.csv' %(yr)).T # rushing yards per game allowed
    opponents = pd.read_csv('opponents_%s.csv' %(yr)).T # the opponents faced for each team
    records = pd.read_csv('records_%s.csv' %(yr)).T # team records (wins: 1, loses: 0)
    
    # this data is from 'teamrankings'
    papg = pd.read_csv('papg_%s.csv' %(yr)) # passing attempets per game
    papga = pd.read_csv('papga_%s.csv' %(yr)) # passing attempts pper game allowed
    qbspg = pd.read_csv('qbspg_%s.csv' %(yr)) # sacks per game
    spg = pd.read_csv('spg_%s.csv' %(yr)) # total sacks per game 
    pcpg = pd.read_csv('pcpg_%s.csv' %(yr)) # passing completions per game
    pcpga = pd.read_csv('pcpga_%s.csv' %(yr)) # passing completions per game allowed
    itpg = pd.read_csv('itpg_%s.csv' %(yr)) # interceptions thrown per game
    irpg = pd.read_csv('irpg_%s.csv' %(yr)) # interceptions received per game
    tapg = pd.read_csv('tapg_%s.csv' %(yr)) # takeaways per game data
    topg = pd.read_csv('topg_%s.csv' %(yr)) # turnovers per game data
    
    # this data is from 'oddssharks'
    lines = pd.read_csv('lines_%s.csv' %(yr)).T # spreads for each game \
    
    # this data is from 'wikipedia'
    psteams = pd.read_csv('PS_Teams_%s.csv' %(yr)) # list of the post season teams
    
    # k = 8 # for 6 games, k=8
    g = k-2 # number of training games
    # training sets
    ppg_train = np.array(ppg.iloc[1:,k-g-2:k-2]) 
    ppga_train = np.array(ppga.iloc[1:,k-g-2:k-2])
    pypg_train = np.array(pypg.iloc[1:,k-g-2:k-2])
    pypga_train = np.array(pypga.iloc[1:,k-g-2:k-2])
    rypg_train = np.array(rypg.iloc[1:,k-g-2:k-2])
    rypga_train = np.array(rypga.iloc[1:,k-g-2:k-2])
    
    # opponents
    opp_train = np.array(opponents.iloc[1:,k-g-2:k-2]) 
    opp_remain = np.array(opponents.iloc[1:,k-2:])
    papg_train = np.array(papg.iloc[:,k-g:k])
    # training sets
    papga_train = np.array(papga.iloc[:,k-g:k])
    qbspg_train = np.array(qbspg.iloc[:,k-g:k])
    spg_train = np.array(spg.iloc[:,k-g:k])
    pcpg_train = np.array(pcpg.iloc[:,k-g:k])
    pcpga_train = np.array(pcpga.iloc[:,k-g:k])
    irpg_train = np.array(irpg.iloc[:,k-g:k])
    itpg_train = np.array(itpg.iloc[:,k-g:k])
    tapg_train = np.array(tapg.iloc[:,k-g:k])
    topg_train = np.array(topg.iloc[:,k-g:k])
    
    # betting spreads
    lines_train = np.array(lines.iloc[1:,k-g-1:k-1])
    records_train = np.array(records.iloc[1:,k-g-2:k-2])
    
    
    def rank(training_set,teams):
        """
        Given a training set of games, this will assign a value to each team based on the stat
    
        Parameters
        ----------
        training_set : nd.array
           Training set for a specific stat
        teams : TYPE
            Array of teams of the same order of the training set
    
        Returns
        -------
        stat_rank : nd.array
            Array with the teams and their stat rank. (unordered)
    
        """
        stat_rank = []
        i = 0
        while i < len(training_set):
            avg = sum(training_set[i,:])/len(training_set[i,:])
            stat_rank.append(avg)
            i += 1
        stat_rank = np.array(stat_rank).reshape(-1,1)
        stat_rank = np.hstack((teams,stat_rank))
        return stat_rank
   
    def librarian(training_set,dictionary,teams):
        """
        Takes in a training set and dictionary and combines the data from the dictionary into an array
    
        Parameters
        ----------
        training_set : nd.array
            Training set for a specific stat
        dictionary : dict
            A dictionary holding the stat values for each team
    
        Returns
        -------
        stat_avgs : nd.array
            An array of the stats and the teams
    
        """
        stat_list = []
        stat_avgs = []
        for row in training_set:
            for team in row:
                stat_list.append(dictionary[team])
            stat = sum(stat_list)/len(stat_list)
            stat_avgs.append(stat)  
        stat_avgs = np.array(stat_avgs).reshape(-1,1)
        stat_avgs = np.hstack((teams,stat_avgs)) 
        return stat_avgs  
        
    
# =============================================================================
# Averages for every team in the respectable stat and gets a league average for that stat
# =============================================================================
    teams = np.array(np.transpose(ppg).columns.values) # gets the team names in a list
    teams = np.delete(teams,0).reshape(-1,1) # deletes the index column header
    
    ppg_avgs = rank(ppg_train,teams)
    ppg_league_avg = sum(ppg_avgs[:,1])/len(ppg_avgs)

    ppga_avgs = rank(ppga_train,teams)
    ppga_league_avg = sum(ppga_avgs[:,1])/len(ppga_avgs)
    
    pypg_avgs = rank(pypg_train,teams)
    pypg_league_avg = sum(pypg_avgs[:,1])/len(pypg_avgs)
    
    pypga_avgs = rank(pypga_train,teams)
    pypga_league_avg = sum(pypga_avgs[:,1])/len(pypga_avgs)
    
    rypg_avgs = rank(rypg_train,teams)
    rypg_league_avg = sum(rypg_avgs[:,1])/len(rypg_avgs)
    
    rypga_avgs = rank(rypga_train,teams)
    rypga_league_avg = sum(rypga_avgs[:,1])/len(rypga_avgs)
    
    papg_avgs = rank(papg_train,teams)
    papg_league_avg = sum(papg_avgs[:,1])/len(papg_avgs)
    
    papga_avgs = rank(papga_train,teams)
    papga_league_avg = sum(papga_avgs[:,1])/len(papga_avgs)

    qbspg_avgs = rank(qbspg_train,teams)
    qbspg_league_avg = sum(qbspg_avgs[:,1])/len(qbspg_avgs)
    
    spg_avgs = rank(spg_train,teams)
    spg_league_avg = sum(spg_avgs[:,1])/len(spg_avgs)
    
    pcpg_avgs = rank(pcpg_train,teams)
    pcpg_league_avg = sum(pcpg_avgs[:,1])/len(pcpg_avgs)

    pcpga_avgs = rank(pcpga_train,teams)
    pcpga_league_avg = sum(pcpga_avgs[:,1])/len(pcpga_avgs)

    irpg_avgs = rank(irpg_train,teams)
    irpg_league_avg = sum(irpg_avgs[:,1])/len(irpg_avgs)
    
    itpg_avgs = rank(itpg_train,teams)
    itpg_league_avg = sum(itpg_avgs[:,1])/len(itpg_avgs)
    
    topg_avgs = rank(topg_train,teams)
    topg_league_avg = sum(topg_avgs[:,1])/len(topg_avgs)
    
    tapg_avgs = rank(tapg_train,teams)
    tapg_league_avg = sum(tapg_avgs[:,1])/len(tapg_avgs)
    
    
# =============================================================================
#     Create a dictionary of the stats that are needed
# =============================================================================
    team_names = list(np.transpose(opponents).columns.values) # gets the team names in a list
    del team_names[0] # deletes the index column header
    ppg_dict = {}
    ppga_dict = {}
    pypg_dict = {}
    pypga_dict = {}
    rypg_dict = {}
    rypga_dict = {}
    papg_dict = {}
    papga_dict = {}
    qbspg_dict = {}
    spg_dict = {}
    topg_dict = {}
    tapg_dict = {}
    i = 0
    while i < len(opp_train):
        ppg_dict[team_names[i]] = ppg_train[i,:]
        ppga_dict[team_names[i]] = ppga_train[i,:]
        pypg_dict[team_names[i]] = pypg_train[i,:]
        pypga_dict[team_names[i]] = pypga_train[i,:]
        rypg_dict[team_names[i]] = rypg_train[i,:]
        rypga_dict[team_names[i]] = rypga_train[i,:]
        papg_dict[team_names[i]] = papg_train[i,:]
        papga_dict[team_names[i]] = papga_train[i,:]
        qbspg_dict[team_names[i]] = qbspg_train[i,:]
        spg_dict[team_names[i]] = spg_train[i,:]
        topg_dict[team_names[i]] = topg_train[i,:]
        tapg_dict[team_names[i]] = tapg_train[i,:]
        
        i += 1
# =============================================================================
#   Adjust the stats
# =============================================================================
    # stats to adjust: pypg, papg, qbspg, itpg, rypg, topg, | spg, pcpg, rypga
    def stat_adjuster(stat_dict,notstat_dict,notstat_league_avg,team_names,opp_train):
        stat_adj = stat_dict
        for row, opponents in enumerate(opp_train):
            for column, opponent in enumerate(opponents):
                stat = stat_dict[team_names[row]][column]
                adjuster = notstat_league_avg/(sum(notstat_dict[opponent])/len(notstat_dict[opponent]))
                stat_adj[team_names[row]][column] = float(stat*adjuster)
        return stat_adj 
          
   
    ppg_adj = stat_adjuster(ppg_dict,ppga_dict,ppga_league_avg,team_names,opp_train)
    ppga_adj = stat_adjuster(ppga_dict,ppg_dict,ppg_league_avg,team_names,opp_train) 
    pypg_adj = stat_adjuster(pypg_dict,pypga_dict,pypga_league_avg,team_names,opp_train) 
    pypga_adj = stat_adjuster(pypga_dict,pypg_dict,pypg_league_avg,team_names,opp_train) 
    rypg_adj = stat_adjuster(rypg_dict,rypga_dict,rypga_league_avg,team_names,opp_train) 
    rypga_adj = stat_adjuster(rypga_dict,rypg_dict,rypg_league_avg,team_names,opp_train) 
    papg_adj = stat_adjuster(papg_dict,papga_dict,papga_league_avg,team_names,opp_train)
    papga_adj = stat_adjuster(papga_dict,papg_dict,papg_league_avg,team_names,opp_train) 
    qbspg_adj = stat_adjuster(qbspg_dict,spg_dict,spg_league_avg,team_names,opp_train)
    spg_adj = stat_adjuster(spg_dict,qbspg_dict,qbspg_league_avg,team_names,opp_train)  
    topg_adj = stat_adjuster(topg_dict,tapg_dict,tapg_league_avg,team_names,opp_train)
    tapg_adj = stat_adjuster(tapg_dict,topg_dict,topg_league_avg,team_names,opp_train)  


        
# =============================================================================
#   Convert Adjusted stats to averages  
# =============================================================================
    def stat_averager(stat_adj):
        stat_avgs = []
        teams = []
        for team in stat_adj:
            avg = sum(stat_adj[team])/len(stat_adj[team])
            stat_avgs.append(avg)
            teams.append(team)
        teams = np.array(teams,dtype = object).reshape(-1,1)
        stat_avgs = np.array(stat_avgs).reshape(-1,1)
        stat_avgs = np.hstack((teams,stat_avgs)) 
        return stat_avgs 
    
    ppg_adj_avgs = stat_averager(ppg_adj)
    ppga_adj_avgs = stat_averager(ppga_adj)
    pypg_adj_avgs = stat_averager(pypg_adj)
    pypga_adj_avgs = stat_averager(pypga_adj)
    rypg_adj_avgs = stat_averager(rypg_adj)
    rypga_adj_avgs = stat_averager(rypga_adj)
    papg_adj_avgs = stat_averager(papg_adj)
    papga_adj_avgs = stat_averager(papga_adj)
    qbspg_adj_avgs = stat_averager(qbspg_adj)
    spg_adj_avgs = stat_averager(spg_adj)
    topg_adj_avgs = stat_averager(topg_adj)
    tapg_adj_avgs = stat_averager(tapg_adj)

    # =============================================================================
    # Strength of team (sot) metric
    #   create a dictionary with the team name as the id and the strength of team calculation as the value
    # =============================================================================
    sot_dict = {} # strength of team dictionary
    
    team_names = list(np.transpose(opponents).columns.values) # gets the team names in a list
    del team_names[0] # deletes the index column header
    
    i = 0
    while i < len(opp_train):
        m1 = ppg_avgs[i,1]
        m2 = ppga_avgs[i,1]
    # =============================================================================
    # Offensive metrics
    # =============================================================================
        net = m1-m2
        o1 = pypg_avgs[i,1]/papg_avgs[i,1] # off. nudger1: passing yards per attempt
        o2 = (qbspg_avgs[i,1]/papg_avgs[i,1]) # off. nudger2: sacks percentage
        o3 = (itpg_avgs[i,1]/papg_avgs[i,1]) # off. nudger3: interception percentage
        o4 = rypg_avgs[i,1]/15 # off. nudger4: rushing yards per game
        o5 = topg_avgs[i,1] # off. nudger5: turnovers per game
        o6 = qbspg_avgs[i,1] # off. nudger6: qb sacks per game
    # =============================================================================
    # Defensive metrics
    # =============================================================================
        d1 = (spg_avgs[i,1]/papga_avgs[i,1]) # def. nudger1: sack percentage
        d2 = pypga_avgs[i,1]/papga_avgs[i,1] # def. nudger5: passing completion percentage
        d3 = rypga_avgs[i,1]/15
    
        a = o1+o4
        b = d2+d3
        c = (d1)/(o3+o5+o6)
        sot_dict[team_names[i]] = math.log((1000*c)**(net*(a/b)),1.5)
            
        # print(sot_dict[team_names[i]])
        i += 1
    
    # The simple strength of team metric
    sot_avgs = []
    teams = []
    for team in sot_dict:
        # print(team)
        # print(sot_dict[team])
        sot_avgs.append(float(sot_dict[team]))
        teams.append(team)
    sot_avgs = np.array(sot_avgs).reshape(-1,1)
    teams = np.array(teams,dtype = object).reshape(-1,1)
    sot_avgs = np.hstack((teams,sot_avgs))
    sot_league_avg = sum(sot_avgs[:,1])/len(sot_avgs[:,1])
    
    # =============================================================================
    # Strength of schedule metrics 
    #   * overall (teams faced so far)
    #   * overall (teams remaining to face)
    # =============================================================================
    teams = np.array(np.transpose(ppg).columns.values) # gets the team names in a list
    teams = np.delete(teams,0).reshape(-1,1) # deletes the index column header
    
    sos_avgs = librarian(opp_train,sot_dict,teams) # strength of opponents faced, overall
    sos_adj_avgs = np.array(sos_avgs[:,1]).reshape(-1,1)
    sos_league_avg = np.mean(sos_adj_avgs)
    sos_adj_avgs = np.hstack((teams,sos_adj_avgs))
    sors_avgs = librarian(opp_remain,sot_dict,teams) # strength of opponents remaining, overall
    sors_adj_avgs = np.array((sors_avgs[:,1] - min(sors_avgs[:,1]))+1).reshape(-1,1)
    sors_adj_avgs = np.hstack((teams,sors_adj_avgs))
    
    # =============================================================================
    # Use the strength of schedule metrics to adjust the strength of team metric
    # =============================================================================
    sot_adj_avgs = []
    
    for i in np.arange(0,len(sos_adj_avgs)):
        sos_adjuster = (sos_adj_avgs[i,1] / sos_league_avg)**(2) # ratio of past to future strength of schedule
        sot_adj_avgs.append(sot_dict[sos_adj_avgs[i,0]]*sos_adjuster) # the first column of row is the teams
    sot_adj_avgs = np.array(sot_adj_avgs).reshape(-1,1)
    sot_adj_avgs = np.hstack((teams,sot_adj_avgs))
    
    # make a dictionary for the adjusted strength of team
    team_names = list(np.transpose(opponents).columns.values) # gets the team names in a list
    del team_names[0] # deletes the index column header
    sot_adj_dict = {}
    i = 0
    while i < len(opp_train):
        sot_adj_dict[team_names[i]] = sot_adj_avgs[i,1]
        i += 1
    
    
    # =============================================================================
    # Get the teams current win and loss totals 
    # =============================================================================
    records_W_L = [] # note: ties are counted as loses
    teams = np.array(topg.iloc[:,1]).reshape(-1,1)
    for row in records_train:
        wins = sum(row)
        loses = len(row)-sum(row)
        
        records_W_L.append([wins,loses])
    records_W_L = np.array(records_W_L).reshape(-1,2)
    records_W_L = np.hstack((teams,records_W_L))
    for index, row in enumerate(opp_train):
        for team in row:
            # print(teams[index,0])
            # print(sot_dict[teams[index,0]])
            # print(sot_dict[team])
            if sot_dict[teams[index,0]] > sot_dict[team]: # team wins
                # print(records_W_L[index])
                records_W_L[index,1] += 1 # the 2nd index in records_W_L is the record, and the 1st index in that is the win total
                # print(records_W_L[index])
            elif sot_dict[teams[index,0]] < sot_dict[team]: 
                # print(records_W_L[index])
                records_W_L[index,2] += 1 # the 2nd index in records_W_L is the record, and the 2nd index in that is the loss total
                # print(records_W_L[index])
                
             
# =============================================================================
# Regression
# =============================================================================
 
    # give the home team a 20% boost and the away team a 20% hit
    # sot_margin_adjusted = np.array([.8*v if location_add[i] == 'home' else .8*v for i,v in enumerate(sot_margin)])
    # training sets
    x_train = np.array(sot_adj_avgs[:,1]).reshape(-1,1)
    point_margin_train_np = np.array(ppg_avgs[:,1] - ppga_avgs[:,1]).reshape(-1,1)
    # line_avgs_add_np = np.array(line_avgs_add).reshape(-1,1)
    y_train = point_margin_train_np
    # y_train = np.array(point_margin_train).reshape(-1,1) # the real spreads are around 3 times as large as the season averages
    # testing sets

    # actual regression
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    from sklearn.preprocessing import PolynomialFeatures
    poly_reg = PolynomialFeatures(degree = 2)
    x_poly = poly_reg.fit_transform(x_train)
    model.fit(x_poly,y_train)

    # prediction plotting data
    x_model_plot = np.arange(min(x_train),max(x_train),.01).reshape(-1,1)
    y_model_plot = model.predict(poly_reg.fit_transform(x_model_plot))

    # plot of the training data
    fig1 = plt.figure(year)
    plt.scatter(x_train,y_train, label = 'Training data')
    plt.plot(x_model_plot,y_model_plot, 'r--', label = 'Regression')
    plt.xlabel('Strength of Team')
    plt.ylabel('Average Point Spread')
    plt.title('%s Trianing Data with Regression' %(year))
    plt.legend()
    plt.show()




    # =============================================================================
    # Sort all the data (for viewing purposes)
    # =============================================================================
    ppg_rank = ppg_avgs[np.argsort(-1*ppg_avgs[:,1])] # points per game: high to low
    ppga_rank = ppga_avgs[np.argsort(ppga_avgs[:,1])] # points per game allowed: low to high
    qbspg_rank = qbspg_avgs[np.argsort(qbspg_avgs[:,1])] # qb sacks per game: low to high
    pypg_rank = pypg_avgs[np.argsort(-1*pypg_avgs[:,1])] # passing yards per game: high to low
    pcpga_rank = pcpga_avgs[np.argsort(pcpga_avgs[:,1])] # passing completions per game against: low to high
    papga_rank = papga_avgs[np.argsort(papga_avgs[:,1])] # passing completions per game against: low to high
    sos_rank = sos_avgs[np.argsort(-1*sos_avgs[:,1])] # strength of schedule: high to low 
    sot_rank = sot_avgs[np.argsort(-1*sot_avgs[:,1])] # strength of team: high to low
    sot_adj_rank = sot_adj_avgs[np.argsort(-1*sot_adj_avgs[:,1])] # strength of team: high to low
    records_W_L_rank = records_W_L[np.argsort(-1*records_W_L[:,1])] # rank teams based on wins: high to low
   
    # =============================================================================
    # A dictionary (league) (nested dictionaries)
    #   * the league contains two elements (type: dict) representing the 2 conferences
    #   * the two conference dictionaries contain 4 elements each (the divisions)
    #   * each of the division elements are assigned str list containing the teams in that division
    # =============================================================================
    league = {} # a dictionary with the 2 conferences 
    AFC = {} # a dictionary of the AFC divisions and teams
    NFC = {} # a dictionary of the NFC divisions and teams
    
    
    AFC['AFC_N']= np.array([['Pittsburgh',0],['Baltimore',0],['Cleveland',0],['Cincinnati',0]],dtype = object)
    AFC['AFC_S'] = np.array([['Houston',0],['Tennessee',0],['Indianapolis',0],['Jacksonville',0]],dtype = object)
    AFC['AFC_E'] = np.array([['New England',0],['Buffalo',0],['NY Jets',0],['Miami',0]],dtype = object)
    AFC['AFC_W'] = np.array([['Kansas City',0],['Denver',0],['Oakland',0],['LA Chargers',0]],dtype = object)
    league['AFC'] = AFC
    
    # print(league['AFC']['AFC_N'])
    NFC['NFC_N'] = np.array([['Green Bay',0],['Minnesota',0],['Chicago',0],['Detroit',0]],dtype = object)
    NFC['NFC_S'] = np.array([['New Orleans',0],['Atlanta',0],['Tampa Bay',0],['Carolina',0]],dtype = object)
    NFC['NFC_E'] = np.array([['Philadelphia',0],['Dallas',0],['NY Giants',0],['Washington',0]],dtype = object)
    NFC['NFC_W'] = np.array([['San Francisco',0],['Seattle',0],['LA Rams',0],['Arizona',0]],dtype = object)
    league['NFC'] = NFC
    
    
    Divisions = ['AFC_N','AFC_S','AFC_E','AFC_W','NFC_N','NFC_S','NFC_E','NFC_W']
    Conferences = ['NFC','AFC'] 
    
    # =============================================================================
    # Obtain the division winners and 2 wild card winners for each conference
    #   * loop through the confereces and divisions 
    #   * rank each team in each division to find division winners
    #   * rank the remaining teams to get the top 2 wild card winners, per conference
    # =============================================================================
    
    # the UNADJUSTED playoff prediction
    print('%s Playoff Predictions' %yr)
    print('traning weeks: %d' %(k-2))
    ps_lst = []
    for conference in league:
        WC = []
        print('\t%s' %(conference))
        for division in league[conference]:
            for index,row in enumerate(league[conference][division]):
                team = row[0] # pulls the team from array
                # print(team)
                for rank,row_test in enumerate(sot_rank):
                    team_test = row_test[0] # this is just the team in the sot_rank array
                    # print(team_test)
                    if team == team_test:
                        # print(rank+1)
                        league[conference][division][index,1] = int(rank+1) # replaces the team with a tuple containing the team and their overall rank
                        # print(league[conference][division])
                        break 
            shorter = league[conference][division] # shortens the next line a bit
            league[conference][division] = shorter[np.argsort(shorter[:,1])] # sorts the teams in the division
            WC.append(league[conference][division][1,:].T) # grab the second place team from the division
            WC.append(league[conference][division][2,:].T) # grab the third place team from the division
            winner = league[conference][division][0,0] # the top team in the division
            ps_lst.append(winner)
            print('%s:\t %s' %(division,winner))
        WC = np.array(WC,dtype = object) 
        WC = WC[np.argsort(WC[:,1])] # sort the potential wild card teams
        WC1 = WC[0,0]
        WC2 = WC[1,0]
        ps_lst.append(WC1)
        ps_lst.append(WC2)
        print('%s_WC1:\t %s' %(conference,WC1))
        print('%s_WC2:\t %s' %(conference,WC2))
    yes = 0
    total = 0
    for team in ps_lst:
        total += 1
        if team in list(psteams.iloc[0,:]):
            yes += 1
    test_stat1 = 100*yes/total
    print('%.0f of the %.0f total teams predicted were correct. (%.2f%%)' %(yes,total,test_stat1))
    
    print('\n')
    # the ADJUSTED playoff prediction     
    print('%s Playoff Predictions - Adjusted' %yr)
    print('traning weeks: %d' %(k-2))
    ps_lst = []
    for conference in league:
        WC = []
        print('\t%s' %(conference))
        for division in league[conference]:
            for index,row in enumerate(league[conference][division]):
                team = row[0] # pulls the team from array
                # print(team)
                for rank,row_test in enumerate(sot_adj_rank):
                    team_test = row_test[0] # this is just the team in the sot_rank array
                    # print(team_test)
                    if team == team_test:
                        # print(rank+1)
                        league[conference][division][index,1] = int(rank+1) # replaces the team with a tuple containing the team and their overall rank
                        # print(league[conference][division])
                        break 
            shorter = league[conference][division] # shortens the next line a bit
            league[conference][division] = shorter[np.argsort(shorter[:,1])] # sorts the teams in the division
            WC.append(league[conference][division][1,:].T) # grab the second place team from the division
            WC.append(league[conference][division][2,:].T) # grab the third place team from the division
            winner = league[conference][division][0,0] # the top team in the division
            ps_lst.append(winner)
            print('%s:\t %s' %(division,winner))
        WC = np.array(WC,dtype = object) 
        WC = WC[np.argsort(WC[:,1])] # sort the potential wild card teams
        WC1 = WC[0,0]
        WC2 = WC[1,0]
        ps_lst.append(WC1)
        ps_lst.append(WC2)
        print('%s_WC1:\t %s' %(conference,WC1))
        print('%s_WC2:\t %s' %(conference,WC2))
    yes = 0
    total = 0
    for team in ps_lst:
        total += 1
        if team in list(psteams.iloc[0,:]):
            yes += 1
    test_stat2 = 100*yes/total
    print('%.0f of the %.0f total teams predicted were correct. (%.2f%%)' %(yes,total,test_stat2))
    return test_stat2

# =============================================================================
#     print('\n')
#     # the ADJUSTED playoff prediction     
#     print('%s Playoff Predictions - by wins' %yr)
#     print('traning weeks: %d' %(k-2))
#     ps_lst = []
#     for conference in league:
#         WC = []
#         print('\t%s' %(conference))
#         for division in league[conference]:
#             for index,row in enumerate(league[conference][division]):
#                 team = row[0] # pulls the team from array
#                 # print(team)
#                 for rank,row_test in enumerate(records_W_L_rank):
#                     team_test = row_test[0] # this is just the team in the array
#                     # print(team_test)
#                     if team == team_test:
#                         # print(rank+1)
#                         league[conference][division][index,1] = int(rank+1) # replaces the team with a tuple containing the team and their overall rank
#                         # print(league[conference][division])
#                         break 
#             shorter = league[conference][division] # shortens the next line a bit
#             league[conference][division] = shorter[np.argsort(shorter[:,1])] # sorts the teams in the division
#             WC.append(league[conference][division][1,:].T) # grab the second place team from the division
#             WC.append(league[conference][division][2,:].T) # grab the third place team from the division
#             winner = league[conference][division][0,0] # the top team in the division
#             ps_lst.append(winner)
#             print('%s:\t %s' %(division,winner))
#         WC = np.array(WC,dtype = object) 
#         WC = WC[np.argsort(WC[:,1])] # sort the potential wild card teams
#         WC1 = WC[0,0]
#         WC2 = WC[1,0]
#         ps_lst.append(WC1)
#         ps_lst.append(WC2)
#         print('%s_WC1:\t %s' %(conference,WC1))
#         print('%s_WC2:\t %s' %(conference,WC2))
#     yes = 0
#     total = 0
#     for team in ps_lst:
#         total += 1
#         if team in list(psteams.iloc[0,:]):
#             yes += 1
#     test_stat3 = 100*yes/total
#     print('%.0f of the %.0f total teams predicted were correct. (%.2f%%)' %(yes,total,test_stat3))
#     return test_stat2
# =============================================================================

# NFL_Model(2017,8)

# =============================================================================
# overall accuracy of the model
# =============================================================================
test_stat = []
k = 8
for year in np.arange(2011,2021):
    test_stat.append(NFL_Model(year,k))
average_accuracy = sum(test_stat)/len(test_stat)
print('\n')
print('Over %d years, using %d training weeks '%(len(test_stat),k-2))
print("\tAverage Model Accuracy: %.2f%%"%average_accuracy)
      



