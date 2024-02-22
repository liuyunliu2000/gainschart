import webbrowser
import PIL
import pandas as pd
import sklearn.metrics as metrics
import numpy as np
from sklearn import  preprocessing
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.tools.tools as tools
import random

import seaborn as sns
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype


def combineimages(list_im, orient='vertical', finame="combined"):
    """
        combine a list of images vertically
            ### combine charts
        list_images = ['image0.png',
                   'image1.png']
        combineimages(list_images)
    """

    imgs    = [ PIL.Image.open(i) for i in list_im ]
    # pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
    min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
    imgs_comb = np.vstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )

    # save picture
    imgs_comb = PIL.Image.fromarray( imgs_comb)
    imgs_comb.save(finame+".png")
    #imgs_comb.show()

def cross_tab_plt(df, ycol='ytarget', collist=[]):
    '''
    for a list of columns: collist. draw a % freq chart

    '''
    for it in collist:
        if df[it].nunique()>10 and is_numeric_dtype(df[it]):
            df[it]=pd.cut(df[it],bins=10,precision=0)

        ttt=pd.crosstab(df[it],df[ycol],normalize='index')
        ttt.plot.bar(stacked=True, figsize=(8, 6))
        plt.title("% of "+ ycol)
        plt.legend(title=ycol, loc='upper left',bbox_to_anchor=(1.04,1))
        for i, txt in enumerate(ttt[1]):
            plt.annotate(round(txt,2),(i-0.1,1))
        plt.show()

def gains_chart(inputdata, truelabel, predlabel, breaks=10, savedimage="gainschart"):
    '''
    Show separation for regresion results for binary y
    inputdata: predicted and actual value
    truelabel: actual value
    predlabel: predicted value
    breaks: # of groups to show separation

    '''
    gainsdata = inputdata.copy()

    #  gainsdata['deciles'] = pd.qcut(gainsdata.loc[:,predlabel], breaks, labels = False, duplicates = 'drop') +1
    gainsdata['deciles'] = np.ceil(gainsdata.loc[:, predlabel].rank(method='first') / float(len(gainsdata)) * breaks)

    avgresp = np.round(gainsdata.loc[:, [truelabel]].mean(), decimals=2)
    gains_tab = pd.DataFrame({"deciles": range(1, breaks + 1)}, index=range(1, breaks + 1))
    gains_tab['decile size'] = gainsdata.loc[:, [predlabel, 'deciles']].groupby('deciles').count()
    gains_tab['min score'] = gainsdata.loc[:, [predlabel, 'deciles']].groupby('deciles').min()
    gains_tab['max score'] = gainsdata.loc[:, [predlabel, 'deciles']].groupby('deciles').max()
    gains_tab['mean score'] = gainsdata.loc[:, [predlabel, 'deciles']].groupby('deciles').mean()
    gains_tab['actual responses'] = gainsdata.loc[:, [truelabel, 'deciles']].groupby('deciles').sum()
    gains_tab['resp rate'] = gains_tab['actual responses'] / gains_tab['decile size']
    gains_tab['gains'] = gains_tab['actual responses'] / sum(gains_tab['actual responses'])
    gains_tab = gains_tab.sort_index(ascending=False)
    gains_tab['cum gains'] = gains_tab['gains'].cumsum()


    ks_tab = gains_tab
    ks_tab['non_resp'] = (ks_tab['decile size'] - ks_tab['actual responses'])
    ks_tab['non_gains'] = ks_tab['non_resp'] / sum(ks_tab['non_resp'])
    ks_tab['cum non gains'] = ks_tab['non_gains'].cumsum()
    ks = (ks_tab['cum gains'] - ks_tab['cum non gains']).max()
    print('KS:', ks)

    roc_auc = metrics.roc_auc_score(gainsdata.loc[:, truelabel], gainsdata.loc[:, predlabel])
    print('ROC AUC:', roc_auc)
    fpr, tpr, _ = metrics.roc_curve(gainsdata.loc[:, truelabel], gainsdata.loc[:, predlabel])

    pr_auc = metrics.average_precision_score(gainsdata.loc[:, truelabel], gainsdata.loc[:, predlabel])
    print('PR AUC:', pr_auc)
    precision, recall, _ = metrics.precision_recall_curve(gainsdata.loc[:, truelabel], gainsdata.loc[:, predlabel])

    gains_tab['roc auc'] = roc_auc
    gains_tab['pr auc'] = pr_auc
    gains_tab['ks'] = ks

    print(gains_tab)
    gains_tab.to_csv("../gaintable/"+savedimage+".csv")

    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label="ROC")
    plt.step(recall, precision, color='b', alpha=0.2, where='post', label="PR_AUC")
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.plot([0, 1], [0, 1], label="random classifier")
    #    plt.xticks(np.arange(0.0,1.0,0.3))
    plt.xlabel('False Positive Rate or Recall')
    plt.ylabel('True Positive Rate or Precision')
    plt.title('%s ROC_AUC %0.2f and PR_AUC %0.2f and KS %0.2f' % (predlabel, roc_auc, pr_auc, ks))
    plt.legend(loc="upper left")

    plt.subplot(1, 2, 2)
    plt.plot(gains_tab['deciles'], gains_tab['resp rate'], label="Resp % by Decile")
    plt.fill_between(gains_tab['deciles'], gains_tab['resp rate'], alpha=0.2, color='b')
    plt.xlabel('Decile')
    plt.ylabel('Resp Rate')
    plt.ylim([0.0, gains_tab.loc[breaks, 'resp rate'] + 0.1])
    plt.xlim([1, breaks])
    plt.plot(gains_tab['deciles'], np.repeat(avgresp, breaks), label='Average Resp %')
    plt.title("%s Gains Chart" % savedimage)
    plt.legend(loc="upper left")

    try:
        plt.savefig("../img/" + savedimage + ".png")
    except:
        print("can not save image")
        pass
    plt.show()


    return gains_tab, [roc_auc, pr_auc, ks]

def gains_chart_bk(inputdata, truelabel, predlabel, breaks=10):
    '''
    Show separation for regresion results for binary y
    inputdata: predicted and actual value
    truelabel: actual value
    predlabel: predicted value
    breaks: # of groups to show separation

    '''
    gainsdata=inputdata.copy()
  #  gainsdata['deciles'] = pd.qcut(gainsdata.loc[:,predlabel], breaks, labels = False, duplicates = 'drop') +1
    gainsdata['deciles']= np.ceil(gainsdata.loc[:,predlabel].rank(method='first')/float(len(gainsdata))*breaks)

    avgresp=np.round(gainsdata.loc[:,[truelabel]].mean(), decimals=2)
    gains_tab = pd.DataFrame({"deciles":range(1,breaks+1)}, index=range(1,breaks+1))
    gains_tab['decile size'] = gainsdata.loc[:,[predlabel,'deciles']].groupby('deciles').count()
    gains_tab['min score'] = gainsdata.loc[:,[predlabel,'deciles']].groupby('deciles').min()
    gains_tab['max score'] = gainsdata.loc[:,[predlabel,'deciles']].groupby('deciles').max()
    gains_tab['mean score'] = gainsdata.loc[:,[predlabel,'deciles']].groupby('deciles').mean()
    gains_tab['actual responses'] = gainsdata.loc[:,[truelabel,'deciles']].groupby('deciles').sum()
    gains_tab['resp rate'] = gains_tab['actual responses']/gains_tab['decile size']
    gains_tab['gains'] = gains_tab['actual responses']/sum(gains_tab['actual responses'])
    gains_tab = gains_tab.sort_index(ascending = False)
    gains_tab['cum gains'] = gains_tab['gains'].cumsum()

    print(gains_tab)

    ks_tab = gains_tab
    ks_tab['non_resp'] = (ks_tab['decile size'] - ks_tab['actual responses'])
    ks_tab['non_gains'] = ks_tab['non_resp']/sum(ks_tab['non_resp'])
    ks_tab['cum non gains'] = ks_tab['non_gains'].cumsum()
    ks = (ks_tab['cum gains'] - ks_tab['cum non gains']).max()
    print('KS:', ks)

    roc_auc = metrics.roc_auc_score(gainsdata.loc[:,truelabel], gainsdata.loc[:,predlabel])
    print('ROC AUC:', roc_auc)
    fpr, tpr, _ = metrics.roc_curve(gainsdata.loc[:,truelabel], gainsdata.loc[:,predlabel])

    pr_auc = metrics.average_precision_score(gainsdata.loc[:,truelabel], gainsdata.loc[:,predlabel])
    print('PR AUC:', pr_auc)
    precision, recall, _ = metrics.precision_recall_curve(gainsdata.loc[:,truelabel], gainsdata.loc[:,predlabel])

    plt.figure(figsize=(16,6))
    plt.subplot(1,2,1)
    plt.plot(fpr,tpr, label="ROC")
    plt.step(recall, precision, color='b', alpha=0.2,where='post' , label="PR_AUC")
    plt.fill_between(recall, precision, step='post', alpha=0.2,color='b')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.plot([0, 1], [0, 1], label="random classifier")
#    plt.xticks(np.arange(0.0,1.0,0.3))
    plt.xlabel('False Positive Rate or Recall')
    plt.ylabel('True Positive Rate or Precision')
    plt.title('%s ROC_AUC %0.2f and PR_AUC %0.2f and KS %0.2f' % (predlabel, roc_auc,pr_auc, ks))
    plt.legend(loc="upper left")

    plt.subplot(1,2,2)
    plt.plot(gains_tab['deciles'], gains_tab['resp rate'], label="Resp % by Decile")
    plt.fill_between(gains_tab['deciles'], gains_tab['resp rate'], alpha=0.2,color='b')
    plt.xlabel('Decile')
    plt.ylabel('Resp Rate')
    plt.ylim([0.0, gains_tab.loc[breaks,'resp rate']+0.1])
    plt.xlim([1 , breaks])
    plt.plot(gains_tab['deciles'],np.repeat(avgresp,breaks), label='Average Resp %')
    plt.title("%s Gains Chart" % predlabel)
    plt.legend(loc="upper left")
    plt.show()
    return  gains_tab
##Usage

#gains_chart(modeldata, dep_var, 'pred', 10)


def gains_continuous(inputdata,  truelabel,  predlabel, breaks =10):
    '''
    Show separation for regresion results for continuous y
    inputdata: dataframe predicted and actual value
    truelabel: actual value
    predlabel: predicted value
    breaks: # of groups to show separation

    '''
    gainsdata=inputdata.copy()
   # gainsdata['deciles'] = pd.qcut(gainsdata.loc[:,predlabel], breaks, labels = False, duplicates = 'raise') +1
    gainsdata['deciles']= np.ceil(gainsdata.loc[:,predlabel].rank(method='first')/float(len(gainsdata))*10.)
    gainsdata['abserror']=abs(gainsdata.loc[:,truelabel]-gainsdata.loc[:,predlabel])

    avgresp=np.round(gainsdata.loc[:,[truelabel]].mean(), decimals=2)
    gains_tab = pd.DataFrame({"deciles":range(1,11)}, index=range(1,11))
    gains_tab['decile size'] = gainsdata.loc[:,[predlabel,'deciles']].groupby('deciles').count()
    gains_tab['min score'] = gainsdata.loc[:,[predlabel,'deciles']].groupby('deciles').min()
    gains_tab['max score'] = gainsdata.loc[:,[predlabel,'deciles']].groupby('deciles').max()
    gains_tab['mean score'] = gainsdata.loc[:,[predlabel,'deciles']].groupby('deciles').mean()
    gains_tab['sum score'] = gainsdata.loc[:,[predlabel,'deciles']].groupby('deciles').sum()
    gains_tab['actual total value'] = gainsdata.loc[:,[truelabel,'deciles']].groupby('deciles').sum()
    gains_tab['actual avg value'] = gainsdata.loc[:,[truelabel,'deciles']].groupby('deciles').mean()
    gains_tab['abs error']        =gainsdata.loc[:,['abserror','deciles']].groupby('deciles').sum()

    gains_tab['gains'] = gains_tab['actual total value']/sum(gains_tab['actual total value'])
    gains_tab = gains_tab.sort_index(ascending = False)
    gains_tab['cum gains'] = gains_tab['gains'].cumsum()
    gains_tab['err %']     =gains_tab['abs error']/gains_tab['actual total value']

    print(gains_tab)
    gains_tab.to_csv("../gaintable/gaintable.csv")

    ks_tab = gains_tab
    ks_tab['counts'] = ks_tab['decile size']/sum(ks_tab['decile size'])
    ks_tab['cum counts'] = ks_tab['counts'].cumsum()
    ks = (ks_tab['cum gains'] - ks_tab['cum counts']).max()
    print('pseudo-KS:', ks)
    #err=round((sum(gains_tab['sum score'])/sum(gains_tab['actual total value'])-1)*100)
    err=round((sum(abs(gainsdata.loc[:,truelabel]-gainsdata.loc[:,predlabel])) / sum(gainsdata.loc[:,truelabel])-1)*100)
    print('error rate (total sum abs(score - actual)/sum actual:',err)
    plt.plot(gains_tab['deciles'], gains_tab['actual avg value'], label="Avg. Actual by Decile")
    plt.plot(gains_tab['deciles'], gains_tab['mean score'], label="Avg. Predicted by Decile" )
    plt.xlabel('Decile')
    plt.ylabel('Predict vs Actual Average Value')
    plt.ylim([0.0, gains_tab[['mean score','actual avg value']].values.max()*1.05])
    plt.xlim([1 , breaks])
    plt.plot(gains_tab['deciles'],np.repeat(avgresp,breaks), label='Overall Actual Average')
    plt.title("Pseudo-Gains Chart for Continuous Y: PS_KS=%0.2f and absolute error=%i %%" % ( ks, err))
    plt.legend(loc="upper left")
    plt.show()
    return  gains_tab
##Usage
## in additon to np.clip()
def clip_price(price,clip):
        price = min(max(clip[0], price), clip[1])
        price = int(round(price))
        return price


######################################################################################################################
### drop high correlation features
######################################################################################################################
def drop_high_correlation_column(df,cutoff=0.7):
    '''
    drop high correlated columns from a df
    output low correlated columns
    '''
    # Create correlation matrix
    corr_matrix_ = df.corr().abs()
    # Select upper triangle of correlation matrix
    upper_ = corr_matrix_.where(np.triu(np.ones(corr_matrix_.shape), k=1).astype(np.bool))

    # Fan found that diagonal is all null so first column is dropped later. so we need set null to 0
    upper_.fillna(value=0, inplace=True)
    # Find index of feature columns with correlation greater than cutoff default 0.7
    #to_drop_ = [column for column in upper_.columns if any(upper_[column] > cutoff)]
    # Drop features
    #df.drop(df.columns[to_drop_], axis=1)
    return [column for column in upper_.columns if any(upper_[column] <= cutoff)]

def excludecolumns(cdl, highfreqdrop=0.99):
    """
    :param cdl:
    :param highfreqdrop:
    :return:
    """
    #### if zero variance feature, delete
    #### if domiance (highest frequency value) is more than highfreqdrop%, delete

    cols=cdl.columns
    sz=cdl.shape[0]
    col_keep=[]

    for i in cols:
        uc=len(pd.Series.unique(cdl[i]))
        vc=pd.Series.value_counts(cdl[i]).sort_values(ascending=False).iloc[0] / sz
        if uc>1 and vc<highfreqdrop:
            col_keep.append(i)

    return cdl[col_keep]

##############################################################################
### runing boostrap to get a recomended list of features to run logistics on
##############################################################################

def bfs(x, y, num_iter, num_feat, topn):
    '''
    import statsmodels.api as sm
    import statsmodels.tools.tools as tools
    import random

    bfs: boostrap feature selection
    x: dataframe with all independent variables, ensure x is all numeric
    y: dataframe with dependent variable, ensure y is unit integer
    num_iter: how many times a small model y= f(x) with a random sample of features is run
    num_feat: how many features are in the samll sample of columns
    topn: how many most likely significant features
    output: list of topn feature
    '''
    # if y.dtype != 'int':
    #    print('y needs to be integer')
    #    y=y.astype('int')
    z = pd.DataFrame()

    #under sampling
    y1i=y[y == 1].index
    y1c=len(y1i)
    y0i=y[y != 1].index
    y0c=min(3*y1c, len(y0i)) ## understand y=0 to be 3x of y=1 or the whole y=1

    cols = list(x)
    for i in range(1, num_iter + 1):
        try:
            sample_i=np.random.choice(y0i, y0c, replace=False)
            x_i = np.concatenate([y1i,sample_i])
            x_column= np.random.choice(x.columns, num_feat, replace=False)
            zz =excludecolumns(x[x_column].loc[x_i])  # some column maybe singular value
            smallmodel = sm.Logit(y.loc[x_i], tools.add_constant(zz)).fit(maxiter=100, disp=0, warn_convergence=False)
            z = z.append(pd.Series.to_frame(smallmodel.pvalues))
        except:
            pass


        if i % 10 == 0:
            print("Loop Number : " + str(i))

    try:
        z.columns = ['pvalue']
        z['feat'] = z.index
        pval_con = z.groupby('feat').mean().sort_values("pvalue")
        pval_con['feat'] = pval_con.index
        pval_con = pval_con[pval_con.feat != 'const']
        pval_con.to_csv("../data/__bfs.csv")

        return pval_con["feat"].head(topn).tolist()
    except:
        print('bfs returned nothing')
        pass

##Usage
#recommended_features = bfs(training_examples, training_targets, 10, 10,30)
def bfs_column_sample(x, y, num_iter, num_feat, topn):
    '''
    import statsmodels.api as sm
    import statsmodels.tools.tools as tools
    import random
    bfs: boostrap feature selection
    x: dataframe with all independent variables, ensure x is all numeric
    y: dataframe with dependent variable, ensure y is unit integer
    num_iter: how many times a small model y= f(x) with a random sample of features is run
    num_feat: how many features are in the samll sample of columns
    topn: how many most likely significant features
    output: list of topn feature
    '''
    # if y.dtype != 'int':
    #    print('y needs to be integer')
    #    y=y.astype('int')
    z = pd.DataFrame()
    random.seed(52)

    cols = list(x)
    for i in range(1, num_iter + 1):
        try:
            smallmodel = sm.Logit(y, tools.add_constant(x.iloc[:, random.sample(range(0, len(cols)), num_feat)])).fit(maxiter=100, disp=0, warn_convergence=False)
            z = z.append(pd.Series.to_frame(smallmodel.pvalues))
        except:
            pass


        if i % 10 == 0:
            print("Loop Number : " + str(i))

    try:
        z.columns = ['pvalue']
        z['feat'] = z.index
        pval_con = z.groupby('feat').mean().sort_values("pvalue")
        pval_con['feat'] = pval_con.index
        pval_con = pval_con[pval_con.feat != 'const']
        pval_con.to_csv("../data/bfs.csv")

        return pval_con["feat"].head(topn).tolist()
    except:
        print('bfs returned nothing')
        pass

def bfs_cont(x, y, num_iter, num_feat, topn):
    '''
    import statsmodels.api as sm
    import statsmodels.tools.tools as tools
    import random

    bfs: boostrap feature selection
    x: dataframe with all independent variables, ensure x is all numeric
    y: dataframe with dependent variable, ensure y is unit integer
    num_iter: how many times a small model y= f(x) with a random sample of features is run
    num_feat: how many features are in the samll sample of columns
    topn: how many most likely significant features
    output: list of topn feature
    '''
    # if y.dtype != 'int':
    #    print('y needs to be integer')
    #    y=y.astype('int')
    z = pd.DataFrame()
    random.seed(52)

    cols = list(x)
    for i in range(1, num_iter + 1):
        try:
            smallmodel = sm.OLS(y, tools.add_constant(x.iloc[:, random.sample(range(0, len(cols)), num_feat)])).fit(method='pinv')
            z = z.append(pd.Series.to_frame(smallmodel.pvalues))
        except:
            pass


        if i % 10 == 0:
            print("Loop Number : " + str(i))

    try:
        z.columns = ['pvalue']
        z['feat'] = z.index
        pval_con = z.groupby('feat').mean().sort_values("pvalue")
        pval_con['feat'] = pval_con.index
        pval_con = pval_con[pval_con.feat != 'const']
        pval_con.to_csv("../data/__bfs.csv")

        return pval_con["feat"].head(topn).tolist()
    except:
        print('bfs returned nothing')
        pass

##Usage
#recommended_features = bfs_cont(training_examples, training_targets, 10, 10,30)

def plot_feature_importance(feature_importances, num_features=25,savedimage='featureimportance'):

    # Write feature importance variables
    with open("./data/"+savedimage+".txt", 'w') as file:
        file.write('Variable \t \t \t \t \t Importance \n')
        for feature in feature_importances:
            file.write('{} \t \t \t \t \t {} \n'.format(feature[0], feature[1]))

    # plot variable importance feature
    fig, ax = plt.subplots()
    feature_top25 = feature_importances[0:num_features]
    # list of x locations for plotting
    x_range = np.arange(len(feature_top25))
    x_vals = [feature[0] for feature in feature_top25]
    y_vals = [feature[1] for feature in feature_top25]
    ax.set_xticks(x_range)
    ax.set_xticklabels(x_vals, rotation='vertical')
    # Make a bar chart
    ax.bar(x_range, y_vals, orientation='vertical', )
    ax.set_ylabel('Importance')
    ax.set_xlabel('Variable')
    ax.set_title('Variable Importances')

    # # Cumulative importances
    # cumulative_importances = np.cumsum(y_vals)
    # ax[1].set_xticklabels(x_vals, rotation='vertical')
    # # Make a line chart
    # ax[1].plot(x_range, cumulative_importances)
    # ax[1].set_ylabel('Cumulative importance')
    # ax[1].set_xlabel('Variable')
    # ax[1].title('Cumulative variable importances')
    # ax[1].hlines(y = 0.8, xmin=0, xmax=len(y_vals), color = 'r', linestyles = 'dashed')
    fileName = "../img/" + savedimage+".png"
    # plt.tight_layout()
    plt.savefig(fileName, bbox_inches='tight')
    plt.show()
    plt.close()


def print_columnname(newcdl,datatype=''):
    '''
    :param dataframe newcdl:
    :return:
    '''
    try:
        for i in range(0,len(newcdl.columns)):
            if datatype:
                if newcdl[newcdl.columns[i]].dtype==datatype:
                    print(newcdl.columns[i])
            else:
               print(newcdl.columns[i])
    except:
        pass

###### data cleaning & drop columns with zero var and too high a top occurance
def read_cleanup(filename="data.csv", highfreqdrop=0.99,removelabel='s3.'):

    cdl=pd.read_csv(filename, header=0, sep=',')

    cdl=cdl.rename(columns=lambda x: x.replace(removelabel,""))

    ##cdl = cdl.reindex(np.random.permutation(cdl.index))
    #### fill na with zero
    #cdl.fillna(value=0, inplace=True)
    for col in cdl.columns:
        if cdl[col].dtype in ['bool','object']:
            print(col,cdl[col].dtype)
            cdl[col]=cdl[col].astype(str)

    #### if zero variance feature, delete
    tt= cdl.apply(lambda x: len(pd.Series.unique(x)))
    cdl=cdl[tt[tt>1].index]
    #cdl=VarianceThreshold(threshold=0).fit_transform(cdl)
    #### if domiance (highest frequency value) is more than highfreqdrop%, delete
    newcdl= cdl.loc[:,cdl.apply(lambda x: pd.Series.value_counts(x).sort_values(ascending=False).iloc[0]/x.size <highfreqdrop).tolist()]

    return newcdl


'''
creat json and hive query for logistic models

'''

def getJSON(mmm,
            modelname='',
            author='',
            parent='',
            modeltype='',
            export='',
            version='',
            isexportonly="true",
            active='',
            scrmultiplier='1.0',
            myquery="null",
            filtr='',
            outfile=''):
      ### model parent and modelname needs populated
      ### lower case and no space and use underscore
      ### author is userid
      ### filter needs a complete where statement
      ### modeltype is either linear or logistic

    i=0
    with open(outfile, "w") as ttt:
        ttt.write("{\n\t\"name\": \""+ modelname+ "\",\n" + \
                  "\t\"author\": \""+ author+ "\",\n" + \
                  "\t\"parent\": \""+ parent+ "\",\n" + \
                  "\t\"model_type\": \""+ modeltype+ "\",\n"+ \
                  "\t\"intercept\": "+ str(mmm[0])+ ",\n"+ \
                  "\t\"export\": \""+ export+ "\",\n"+ \
                  "\t\"version\": "+ version+ ",\n"+ \
                  "\t\"is_export_only\": "+ isexportonly + ",\n"+ \
                  "\t\"final_score_multiplication_factor\": "+ scrmultiplier+ ",\n"+ \
                  "\t\"myquery\": "+ myquery+ ",\n"+ \
                  "\t\"active\": "+ active+ ",\n"+ \
                  "\t\"query_filter\": \""+ filtr+ "\",\n"+ \
                  "\t\"parms\": {\n" )
        for i in range(1,len(mmm)):
            ttt.write("\t\t\"feature_" + str(i) + "\": {\n" + \
                        "\t\t\t\"parm\": "+ str(mmm[i])+",\n" + \
                        "\t\t\t\"feat_name\": \"coalesce("+ mmm.index[i].replace(" ", "_") +",0)\",\n"+ \
                        "\t\t\t\"interactions\": null\n")
            if i == len(mmm)-1:
                ttt.write("\t\t}\n\t}\n}")
            else:
                ttt.write("\t\t},\n")

#
#mmm=coachmodel.params
#getJSON(mmm,   ##This is the model object directly out of logistic regression
#         modelname = "internal_hosting_migration",
#         author = "yyliu",
#         parent = "propensity",
#         modeltype = "logistic",
#         export = "teradata",
#         version = "1.0",
#         active = "true",
#         myquery="null",
#         isexportonly="TRUE",
#         scrmultiplier="1",
#         filtr = "",
#         outfile = "../cards/internal_migration.json")


##########################################################
### regression score cards
##########################################################
def scrcard(mmm,
            modelname="regression model",
            scrfile="../cards/internal_migration.sql"):
    with open(scrfile, "w") as ttt:
        ttt.write("--- " + modelname + "\n 1/(1+exp(-1*(\n")
        for i in range(0, len(mmm)):
            if i==0:
                ttt.write(str(mmm[i])+ "\n")
            else:
                ttt.write('{0:+.5f}'.format(mmm[i]) + "*coalesce(" + mmm.index[i].replace(" ","_") + ",0) \n")
        ttt.write(" ))) \n as " + modelname + " \n ,")
#scrcard(mmm, modelname="regression_model", scrfile="../cards/internal_migration.sql")
def scrcard_insert_model_score(modelname="regression model",
            scrfile="../cards/insert_example.ins"):
    with open(scrfile, "w") as ttt:
        ttt.write("---insert  " + modelname + " into bi.ba_marketing.model_scores \n\n")
        ttt.write("INSERT into bi.ba_marketing.model_scores \n")
        ttt.write("select \n")
        ttt.write("shopper_id , -- cannot be null  \n")
        ttt.write("NULL as resource_id , -- can be null  \n")
        ttt.write("NULL as alternate_id , -- can be null  \n")
        ttt.write("'"+ modelname + "' as model , \n")
        ttt.write("'propensity' as model_parent ,  \n")
        ttt.write("'1.0' as version ,  \n")
        ttt.write("'logistic' as model_type ,   \n")
        ttt.write("current_date as dt ,   \n")
        ttt.write("NULL as var1 , -- can be null  \n")
        ttt.write("NULL as var2 , -- can be null   \n")
        ttt.write("NULL as logit , -- can be null   \n")
        ttt.write( modelname +" as score  -- can not be null (must be bounded between 0 and 1) \n" )
        ttt.write("from \n")
        ttt.write("model_score_tmp ; \n")

#scrcard_insert_model_score(modelname="regression_model", scrfile="../cards/insert_example.ins")

def scrcard_reg(mmm,
            modelname="regression model",
            scrfile="../cards/internal_migration.sql"):
    with open(scrfile, "w") as ttt:
        ttt.write("--- " + modelname + "\n")
        for i in range(0, len(mmm)):
            if i==0:
                ttt.write(str(mmm[i])+ "\n")
            else:
                ttt.write('{0:+}'.format(mmm[i]) + "*coalesce(" + mmm.index[i].replace(" ","_") + ",0) \n")
        ttt.write(" \n as" + modelname + " \n ,")
#scrcard_reg(mmm, modelname="regression_model", scrfile="../cards/internal_migration.sql")


class Number(object):

    def __init__(self, n):
        self.value = n

    def val(self):
        return self.value

    def add(self, n2):
        self.value += n2.val()

    def __add__(self, n2):
        return self.__class__(self.value + n2.val())

    def __str__(self):
        return str(self.val())

    @classmethod
    def addall(cls, number_obj_iter):
        cls(sum(n.val() for n in number_obj_iter))
