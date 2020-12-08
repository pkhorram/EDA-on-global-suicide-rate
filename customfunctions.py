import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def change_gdp_to_ints(df):
    '''
    convert gdp_for_year from string to ints 
    '''

    gdp = []
    # convert gdp_for_year from string to ints 
    for i, string in enumerate(df[df.columns[8]]):
        gdp.append(int(string.replace(',','')))

    # put the results in a new column    
    gdp_frame =pd.DataFrame()
    gdp_frame['gdp'] = np.array(gdp)
    new_df = pd.concat((df,gdp_frame),axis=1)

    return new_df

def add_conts(df):
    '''
    adds continent column
    '''
    cont = pd.read_csv('./countryContinent.csv',encoding = "ISO-8859-1")
    
    l = []
    count_list = []
    for country in df['country']:
        if country not in count_list:
            count_list.append(country)
            if country == 'Saint Vincent and Grenadines':
                country = 'Saint Vincent and the Grenadines'
            elif country == 'United Kingdom':
                country = 'United Kingdom of Great Britain and Northern Ireland'
            elif country == 'United States':
                country = 'United States of America'
            elif country == 'Macau':
                country = 'Macao'
            elif country == 'Republic of Korea':
                country = "Korea (Democratic People's Republic of)"
        
            l.append(cont[cont['country'] == country].continent.to_list()[0])

    cont_dict = {}         
    for i in range(len(count_list)):   
        cont_dict[count_list[i]] = l[i]


    new_col = pd.DataFrame()
    new_col['continent'] = df['country']
    new_col = new_col.replace({"continent": cont_dict})

    new_df = pd.concat((new_col, df), axis=1)

    return new_df


def remove_HDI(df):
    '''
    removes 'HDI for year' column from given df
    '''
    new_df = df.drop(['HDI for year'], axis =1)
    new_df = new_df.dropna()
    return new_df

def rename_suicide_rate(df):
    '''
    rename 'suicides/100k pop' to 'suicides_per_100k'
    '''
    new_df=df.rename(columns={"suicides/100k pop": "suicides_per_100k"})
    return new_df

def remove_2016(df):
    '''
    removes rows from 2016
    '''
    new_df=df[df.year != 2016]
    return new_df


def preprocess(df):
    '''
    prepares the dataframe
    '''
    new_df = remove_HDI(df)
    new_df = change_gdp_to_ints(new_df)
    new_df = rename_suicide_rate(new_df)
    new_df = remove_2016(new_df)
    new_df = add_conts(new_df)
    return new_df

def suiciderate_gender_year(df):
    '''
    plots suicide rate of each gender over the years
    '''
    for gender in df['sex'].unique():
        df2 = df[df['sex'] == gender]
        dict = {}
        count =  np.array(df2['suicides_per_100k'])
        for ind, y in enumerate(df2['year']):
            if y not in dict:
                dict[y] = 0
                dict[y] += count[ind]
            else:
                dict[y] += count[ind]
                
        lists = list(dict.items())
        x,y = zip(*lists)
        plt.bar(x,y, label=gender)

        plt.legend(loc='best') 
    
    plt.title('Suicide Rate for each sex over different Years')
    plt.xlabel('Year')
    plt.ylabel('counts/100k')
    plt.grid(True)
    plt.show()

def suicidecount_gender_year(df):
    '''
    plots suicide count of each gender over the years
    '''
    for gender in df['sex'].unique():
        df2 = df[df['sex'] == gender]
        dict = {}
        count =  np.array(df2['suicides_no'])
        for ind, y in enumerate(df2['year']):
            if y not in dict:
                dict[y] = 0
                dict[y] += count[ind]
            else:
                dict[y] += count[ind]
                
        lists = list(dict.items())
        x,y = zip(*lists)
        plt.bar(x,y, label=gender)

        plt.legend(loc='best') 
    
    plt.title('Suicide Counts for each sex over different Years')
    plt.xlabel('Year')
    plt.ylabel('counts')
    plt.grid(True)
    plt.show()

def suiciderate_gdp_gender(df, country='worldwide'):

    if country=='worldwide':
        p1 = df
    else:
        p1 = df[df['country'] == country]


    for gender in p1['sex'].unique():
        pm = p1[p1['sex'] == gender]
        sui_num = []
        gdp = []
        for year in pm['year'].unique():
            p2 = pm[pm['year'] == year]
            sui_num.append(sum(p2.suicides_per_100k))
            gdp.append(np.average(p2['gdp']))
            
        x = []
        y = []
        for i,j in sorted(zip(gdp,sui_num)):
            x.append(i)
            y.append(j)

        plt.scatter(x,y,label=gender)
    
    
    plt.legend(loc='best')
    
    plt.title('{}'.format(country))
    plt.xlabel('gdp')
    plt.ylabel('suicide count/100k')
    plt.grid()
    plt.show()

def suicidecount_gdp_gender(df, country='worldwide'):

    if country=='worldwide':
        p1 = df
    else:
        p1 = df[df['country'] == country]

    for gender in p1['sex'].unique():
        pm = p1[p1['sex'] == gender]
        sui_num = []
        gdp = []
        for year in pm['year'].unique():
            p2 = pm[pm['year'] == year]
            sui_num.append(sum(p2.suicides_no))
            gdp.append(np.average(p2['gdp']))
            
        x = []
        y = []
        for i,j in sorted(zip(gdp,sui_num)):
            x.append(i)
            y.append(j)

        plt.scatter(x,y,label=gender)
    
    plt.legend(loc='best')
    
    plt.title('{}'.format(country))
    plt.xlabel('gdp')
    plt.ylabel('suicide count')
    plt.grid()
    plt.show()



def suiciderate_age_year(df):

    for age in df['age'].unique():
        df2 = df[df['age'] == age]


        for gender in df['sex'].unique():
            dfm = df2[df2['sex']== gender]
            dict = {}
            count =  np.array(dfm['suicides_per_100k'])
            for ind, y in enumerate(dfm['year']):
                if y not in dict:
                    dict[y] = 0
                    dict[y] += count[ind]
                else:
                    dict[y] += count[ind]
                    
            
            lists = list(dict.items())
            x,y = zip(*lists)
            plt.bar(x,y, label=gender)
        
        
        plt.legend(loc='best')

        plt.title('age: {}'.format(age))
        plt.xlabel('year')
        plt.ylabel('suicides/100k pop')
        plt.grid()
        plt.show()

def suiciderate_age(df):
    ages=[]
    rate=[]
    
    for age in df['age'].unique():
        ages.append(age)
        rate.append(np.sum(df[df['age']==age].suicides_per_100k))
        
    
    fig = plt.figure(figsize=(15,10))
    plt.pie(rate, labels=ages, autopct='%1.1f%%')
    plt.title('suicides rate population by age')
    plt.show()

def suicidecount_age(df):
    ages=[]
    rate=[]
    
    for age in df['age'].unique():
        ages.append(age)
        rate.append(np.sum(df[df['age']==age].suicides_no))
        
    
    fig = plt.figure(figsize=(15,10))
    plt.pie(rate, labels=ages, autopct='%1.1f%%')
    plt.title('suicide number total by age')
    plt.show()

def suiciderate_cont_time(df):
    
    for cont in df['continent'].unique():
        p1 = df[df['continent']==cont]
        dic ={}
        count = np.array(p1['suicides_per_100k'])
        for ind, y in enumerate(p1['year']):
            if y not in dic:
                dic[y]=0
                dic[y]+=count[ind]
            else:
                dic[y]+= count[ind]
                
                
        lists = list(dic.items())
        
        x,y=zip(*lists)
        
        plt.scatter(x,y,label=cont,marker='.')
        
    plt.legend(loc='best')
    
    plt.title('Suicide Rate by Continent of Different Years')
    plt.xlabel('Year')
    plt.ylabel('counts/100k population')
    plt.grid(True)
    plt.show()

def suiciderate_country_time(df,country_list):
    for country in country_list:
        p1 = df[df['country']==country]
        dic ={}
        count = np.array(p1['suicides_per_100k'])
        for ind, y in enumerate(p1['year']):
            if y not in dic:
                dic[y]=0
                dic[y]+=count[ind]
            else:
                dic[y]+= count[ind]
                
                
        lists = list(dic.items())
        
        x,y=zip(*lists)
        
        plt.scatter(x,y,label=country)
    
    plt.legend(loc='best')
    
    plt.title('Suicide Rate by Country of Different Years')
    plt.xlabel('Year')
    plt.ylabel('counts/100k population')
    plt.grid(True)
    plt.show()

def suicides_cont_avg(df):
    for cont in df['continent'].unique():
        p1 = df[df['continent']==cont]
        cont_max_avg={cont:{'temp':0}}
        cont_min_avg={cont:{'temp':float('inf')}}
        for country in p1['country'].unique():
            p2 = p1[p1['country']==country]
            dic ={}
            count = np.array(p2['suicides_per_100k'])
            for ind, y in enumerate(p2['year']):
                if y not in dic:
                    dic[y]=0
                    dic[y]+=count[ind]
                else:
                    dic[y]+= count[ind]
                
            lists = np.array(list(dic.values()))
            
            if np.mean(lists)>list(cont_max_avg[cont].values())[0]:
                cont_max_avg={cont:{country:np.mean(lists)}}
            if np.mean(lists)<list(cont_min_avg[cont].values())[0] and np.mean(lists)>0:
                cont_min_avg={cont:{country:np.mean(lists)}}
                
        print('Continent: '+str(list(cont_max_avg.keys())[0])+'\n Highest rate: '+str(list(cont_max_avg.values()))+'\n Lowest rate: '+str(list(cont_min_avg.values())))

        
def suicide_pearson_population(df):
    
    country_list=np.unique(df['country'])
    year_list=list(np.unique(df['year']))

    year_list.remove(1985)
    year_list.remove(1986)
    year_list.remove(2016)
        
    pear=[]
    
    for co in country_list:
        
        df_c = df[df['country'] == co]
       
        dff_facto=[]
        dff_sui=[]
        
        for year in year_list:
            
            df_y = df_c[df_c['year'] == year]
            df_y_s=list(df_y['suicides_no'])
            n=sum(df_y_s)
            
            dff_sui.append(n)
            
            df_y_f=list(df_y['population'])
            df_y_f_int=[]
            
            df_y_f_int=df_y_f
                        
            nn=sum(df_y_f_int)
     
            dff_facto.append(nn)            
        
        cor=np.corrcoef(dff_sui,dff_facto)
        pear.append(cor[1][0])

    return pear
        
