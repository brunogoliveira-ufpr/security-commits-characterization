import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from numpy import std,mean
from math import sqrt
import researchpy
#from cliffs_delta import cliffs_delta                                                                                                                                                                                                     
                                                                                                                                                                                                                                           
                                                                                                                                                                                                                                           
pd.set_option("display.max_rows", None, "display.max_columns", None)                                                                                                                                                                       
df = pd.read_csv('[final]hermes_all_metrics.csv', header=0)                                                                                                                                                                                
#print(df.columns.tolist())                                                                                                                                                                                                                
#df = pd.DataFrame(df)                                                                                                                                                                                                                     
#df = df[(df[['modifications','complexity']] != 0).all(1)]                                                                                                                                                                                 
#df = df[df.modified_file_path.str.contains(r'^.*\.(cpp|h)$')]                                                                                                                                                                             
#df = (df[(df['file_name'].str.contains(r'^.*\.(cpp|c|cc|h)$'))])                                                                                                                                                                          
#f = open("output_csv.txt", "w+")                                                                                                                                                                                                          
#df = (df[(df['AvgLine'].notnull())])                                                                                                                                                                                                      
#df = (df[(df['commit_author'].str.contains(r'^.*(@microsoft.com)$'))])                                                                                                                                                                    
df3 = (df[(df['type'].str.contains('security'))])                                                                                                                                                                                          
df2 = (df[(df['type'].str.contains('other'))])                                                                                                                                                                                             
                                                                                                                                                                                                                                           
def cliffs_delta(control, test):                                                                                                                                                                                                           
    """                                                                                                                                                                                                                                    
    Computes Cliff's delta for 2 samples.                                                                                                                                                                                                  
    See https://en.wikipedia.org/wiki/Effect_size#Effect_size_for_ordinal_data                                                                                                                                                             
                                                                                                                                                                                                                                           
    Keywords                                                                                                                                                                                                                               
    --------                                                                                                                                                                                                                               
    control, test: numeric iterables.                                                                                                                                                                                                      
        These can be lists, tuples, or arrays of numeric types.                                                                                                                                                                            
                                                                                                                                                                                                                                           
    Returns                                                                                                                                                                                                                                
    -------                                                                                                                                                                                                                                
        A single numeric float.                                                                                                                                                                                                            
    """                                                                                                                                                                                                                                    
    import numpy as np                                                                                                                                                                                                                     
    from scipy.stats import mannwhitneyu                                                                                                                                                                                                   
                                                                                                                                                                                                                                           
    # Convert to numpy arrays for speed.                                                                                                                                                                                                   
    # NaNs are automatically dropped.                                                                                                                                                                                                      
    if control.__class__ != np.ndarray:                                                                                                                                                                                                    
        control = np.array(control)                                                                                                                                                                                                        
    if test.__class__ != np.ndarray:                                                                                                                                                                                                       
        test    = np.array(test)                                                                                                                                                                                                           
                                                                                                                                                                                                                                           
    c = control[~np.isnan(control)]                                                                                                                                                                                                        
    t = test[~np.isnan(test)]                                                                                                                                                                                                              
                                                                                                                                                                                                                                           
    control_n = len(c)
    test_n = len(t)

    # Note the order of the control and test arrays.
    U, _ = mannwhitneyu(t, c, alternative='two-sided')
    cliffs_delta = ((2 * U) / (control_n * test_n)) - 1

    # more = 0
    # less = 0
    #
    # for i, c in enumerate(control):
    #     for j, t in enumerate(test):
    #         if t > c:
    #             more += 1
    #         elif t < c:
    #             less += 1
    #
    # cliffs_delta = (more - less) / (control_n * test_n)

    return cliffs_delta


def cohen_d(x,y):
        return (mean(x) - mean(y)) / sqrt((std(x, ddof=1) ** 2 + std(y, ddof=1) ** 2) / 2.0)

for col in df2.columns:
        if(col != 'project' and col != 'type' and col != 'commit_hash' and col != 'file_name' and col != 'modified_file_path' and col != 'commit_author'):
        #if(col == 'lines_added'):
                df4 = df3[col].groupby(df3['commit_hash']).mean()
                df5 = df2[col].groupby(df2['commit_hash']).mean()
                #print(len(df4)+len(df5))
#                print(researchpy.ttest(df5, df4, paired=False, equal_variances=False))
                stat,p = stats.mannwhitneyu(df4,df5)
                d = cliffs_delta(df4,df5)
                print('Hermes, %s, %.10f, %.10f, %.3f, %.3f' % (col,stat, p, cohen_d(df4,df5), d))
#                df_result1, df_result2 = researchpy.ttest(df5, df4, paired=False, equal_variances=False)
#                print(df_result2.iloc[[7]])
                #if (col.len() < 20):
                data = {'security':df4,'other':df5}
                df = pd.DataFrame(data=data)
                df.boxplot(column=['other','security'])
                if (len(col) < 20):
                  plt.ylabel('{}'.format(col))
                  plt.title('{} Mean Value'.format(col))
                  plt.savefig('{}_hermes.png'.format(col))
                else:
                  plt.ylabel('any')
                  plt.title('any')
                  plt.savefig('any.png')
#print(df.file_name.describe(include='all'))
#sns.set_theme(style="white")
#sns.pairplot(data=df, hue="file_name")

#print(df.mean(skipna = True).to_csv())
#print(df.nunique().to_csv())
#print(df.mean().to_csv())
#df = (df[(df['Kind'].str.contains('File'))])
#f.write(str(df.sum(skipna = True)))
#f.close()
#df.to_csv('1ffa6cf91e5ce62d1d8fd84e6c77a4d2f1227466-new.udb.csv', index=False)
