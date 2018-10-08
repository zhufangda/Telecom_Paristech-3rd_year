import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

def plot_lift(y_true,soft_pred):
    df = pd.DataFrame(np.array([y_true,soft_pred]).T,columns=['true','soft'])
    df.sort_values(by='soft',ascending=False,inplace=True)
    lift = pd.concat([df.groupby('soft').count(),df.groupby('soft').sum()],axis=1)
    lift.sort_index(ascending=False,inplace=True)
    lift.columns = ['strate size','positive']
    lift['negative'] = lift['strate size'] - lift['positive']
    n_positive = df['true'].sum()
    n_negative = df['true'].count()-n_positive
    lift.index.name = 'threshold'
    lift['Group size'] = lift['strate size'].cumsum()
    lift['Group hit probability'] = lift['positive'].cumsum() / lift['Group size']
    lift['Group ratio'] = lift['Group size'] / len(y_true)
    lift['Lift'] = lift['Group hit probability'] / (n_positive /(n_positive+n_negative))
    plt.plot(lift['Group ratio'].values,lift['Lift'].values)
    plt.xlim((0.1, 1))
    plt.grid()
    return 