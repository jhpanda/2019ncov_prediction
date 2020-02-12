
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lmfit import Minimizer, Parameters, fit_report

def lmfunc(pars, x, y=None):
    a,b,c = pars['a'],pars['b'],pars['c']
    #model = a/(1+np.exp(-b*x))+c
    #model = a/(1+np.exp(-b*x))
    model = a*(c/a)**(np.exp(-b*x))
    if y is None:
        return model
    res = model-y
    return res

def growth(a,b,c,x):
    model = a*(c/a)**(np.exp(-b*x))
    return model

def lmsig(pars,x,y=None):
    #a,b,c,d = pars['a'],pars['b'],pars['c'],pars['d']
    a,b,c = pars['a'],pars['b'],pars['c']
    model = a/(1+np.exp(-b*(x-c)))
    if y is None:
        return model
    res = model-y
    return res

def sigm(a,b,c,x):
    model = a/(1+np.exp(-b*(x-c)))
    return model

## this is useful when we want to use differential prediction
def lmdfunc(pars, x, y=None):
    a,b = pars['a'], pars['b']
    v = 1/(1+np.exp(-b*x))
    #da = 1-exp_v
    #db = a*x*exp_v
    da = v
    db = a*x*v
    dc = np.ones(len(x))
    return np.array([da,db,dc])

def fit(x_tofit,y_tofit,func='Growth',daystofit=60,nfreq=7):

    y0 = y_tofit[0]
    y1 = y_tofit[-1]
    x1 = x_tofit[-1]
    pars = Parameters()
    pars.add('a', value=y1,min=y1)
    pars.add('b', value=0.,min=0.)
    pars.add('c', value=y0,min=0.)
  
    if func=='Growth':
        lfunc = lmfunc
        growf = growth
    elif func=='Sigmoid':
        lfunc = lmsig
        growf = sigm
    mfit = Minimizer(lfunc,pars,fcn_args=(x_tofit,),fcn_kws={'y':y_tofit},reduce_fcn='neglogcauchy')
    nfit = mfit.minimize(method='nedeler')
    dfit = mfit.minimize(method='leastsq',params=nfit.params)

    a = dfit.params['a']
    b = dfit.params['b']
    c = dfit.params['c']

    ## to look ahead
    nstd  = 3
    a_err = nstd*a.stderr
    b_err = nstd*b.stderr
    c_err = nstd*c.stderr

    #print(fit_report(dfit))
    
    xfit = np.arange(0,daystofit,1.)
    yfit = lfunc(dfit.params, xfit)

    fig,ax = plt.subplots()

    #yfit1 = growf(a+a_err,b-b_err,c+c_err,xfit)
    #yfit2 = growf(a-a_err,b+b_err,c-c_err,xfit)
    yfit1 = growf(a+a_err,b,c,xfit)
    yfit2 = growf(a-a_err,b,c,xfit)
    #plt.fill_between(xfit, yfit2,yfit1, color="#ABABAB")
    plt.fill_between(xfit, yfit2,yfit1, color="#ABABAB",label='CI: 99.75%')

    ax.plot(xfit,yfit,'r-',label="Best fit")
    ax.scatter(x_tofit,y_tofit,c='black',s=50,label="# Diagnosed in Mainland China")

    #xlabel = [date[s].date() for s in range(daystofit) if s%7==0]
    #print(date.astype(str)[0][5:])
   
    nfreq = 3
    date = pd.date_range('2020-01-16',periods=daystofit)
    datestr = date.astype(str)
    ypred = lfunc(dfit.params,x1+1)
    yerr  = a.stderr/a
    ypred_1 = ypred * (1+1.732*yerr)
    ypred_2 = ypred * (1-1.732*yerr)
    print("Predicted diagnosed number on %s: %d (total: %d-%d, 95%% CI)"%(datestr[x1+1],ypred-y1,ypred_2,ypred_1))
    xticks = [xfit[s] for s in range(daystofit) if s%nfreq==0]
    xlabel = [datestr[s][5:] for s in range(daystofit) if s%nfreq==0]
    plt.xticks(xticks,xlabel,rotation=60)

    plt.legend()
    fig.tight_layout()
    plt.show()
    fig.savefig('FitbyGompertzGrowth.pdf')
    plt.close()

def import_source(source='wiki.dat'):
    days = []
    data = []
    with open(source,'r') as p:
        lines = p.readlines()
        for line in lines:
            d1,d2 = line.split()
            days += [int(d1)]
            data += [int(d2)]
    return np.array(days),np.array(data)

if __name__ == '__main__':
    data = [45,62,121,198,291,440,571,830,1287,1975,2744,4515,5974,7711,9692,11791,14380,17205,20438,24324,28018,31161,34546,37198,40171]
    #data = np.array(data)
    #days = np.arange(0,len(data),1.)
    nfreq     = 3
    daystofit = 75
    func      = 'Growth'
    days,data = import_source()
    #days = days[0:-1]
    #data = data[0:-1]
    fit(days,data,func=func,daystofit=daystofit,nfreq=nfreq)
