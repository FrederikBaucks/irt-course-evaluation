import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import os

# Get parent directory of current file
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#Get parent dir of parent dir
parent_dir = os.path.dirname(parent_dir)
MISSING_VALUE = -99999

def plot_validity(degree, model, verbose=False, save_path=''):
    '''
        
    '''
    data_folder = parent_dir + '/data/real/' + degree + '/'

    temp_model = ''
    # translate model name:
    if "mmod1" in model:
        temp_model = "1pl"
    else:
        temp_model = "2pl"
    
    if "dim1" in model:
        temp_model += "_1dim"
        dim = 1
    elif "dim2" in model:
        temp_model += "_2dim"
        dim = 2
    elif "dim3" in model:
        temp_model += "_3dim"
        dim = 3
    else:
        temp_model += "_1dim"
        dim=1
    #dim=3
    #temp_model = "2pl_3dim"
    # Load the data
    items = np.array(pd.read_csv(data_folder + "binary_reduced/item_ids.csv", header=None)).flatten()
    
    p_r = np.array(pd.read_csv(data_folder +  "binary_reduced/pass_rates.csv", header=None))[1:,1].astype(float)
    
    
    d_1pl = np.array(pd.read_csv(data_folder + "irt_results/"+ temp_model +"/diff.csv"))
    
    gpa = np.array(pd.read_csv(data_folder + "non_binary/gpas.csv", header=None))
    ids_non_bin = gpa[1:,0]
    gpa = gpa[1:,1].astype(float)
    
    gpa_bin = np.array(pd.read_csv(data_folder + "binary_reduced/gpas.csv", header=None))
    ids_binary = gpa_bin[1:,0]

    abi_1pl = np.array(pd.read_csv(data_folder + "irt_results/"+ temp_model +"/abilities.csv"))
    print(abi_1pl.shape)

    #copy gpa 
    gpa_copy = gpa.copy()

    #construct array with indexes of student ids that are in both datasets
    idx = np.array([np.where(ids_non_bin == i)[0][0] for i in ids_binary])

    #delete entries from gpa_copy that are not in both datasets
    gpa_copy = gpa_copy[idx]


    

    #Dont show plots if verbose is false
    if not verbose:
        plt.ioff()
    else:
        plt.ion()


    
    # covid_20=[]
    # rest =[]
    # for i, item  in enumerate(items):
    #     if ('20' in item and '19' not in item)   or '21' in item:
    #         covid_20.append(i)
    #     else:
    #         rest.append(i)
    # covid_20=np.array(covid_20)
    rest= np.arange(len(items))#np.array(rest)
    
    if dim == 2:
        # load loadings 
        loadings = pd.read_csv(data_folder + "irt_results/models/model_loadings/mmmod2_dim2.txt", header=None)
        # get last row
        loadings = loadings.iloc[-1,:].values
        # seperate string by space
        loadings = loadings[0].split(" ")[3:5]
        # convert to float
        loadings = np.array([float(i) for i in loadings])
        d_1pl = (d_1pl[:,0])#*loadings[0] + d_1pl[:,1]*loadings[1])*(1/(loadings[0]+loadings[1]))
        abi_1pl = (abi_1pl[:,0]*loadings[0] + abi_1pl[:,1]*loadings[1])*(1/(loadings[0]+loadings[1]))
    elif dim == 3:
        # load loadings
        loadings = pd.read_csv(data_folder + "irt_results/models/model_loadings/mmmod2_dim3.txt", header=None)
        # get last row
        loadings = loadings.iloc[-1,:].values
        # seperate string by space
        loadings = loadings[0].split(" ")[3:6]
        # convert to float
        loadings = np.array([float(i) for i in loadings])
        d_1pl = (d_1pl[:,0])#*loadings[0] + d_1pl[:,1]*loadings[1] + d_1pl[:,2]*loadings[2])*(1/(loadings[0]+loadings[1]+loadings[2]))
        abi_1pl = (abi_1pl[:,0]*loadings[0] + abi_1pl[:,1]*loadings[1] + abi_1pl[:,2]*loadings[2])*(1/(loadings[0]+loadings[1]+loadings[2]))

    # set inf values to nan
    d_1pl[d_1pl == np.inf] = np.nan
    d_1pl[d_1pl == -np.inf] = np.nan
    # identify nan values in d_1pl
    nan_idx = np.argwhere(np.isnan(d_1pl))
    # delete nan values from d_1pl and p_r
    d_1pl = np.delete(d_1pl, nan_idx)
    p_r = np.delete(p_r, nan_idx)
    rest = np.delete(rest, nan_idx)
    # warn user about nan values
    if nan_idx.size > 0:
        print("Warning: There are nan values in the difficulty array. These values have been removed from the plot.")

    # get index of d_1pl values that are abs()>10
    idx = np.argwhere(np.abs(d_1pl)>10)
    # delete values from d_1pl, p_r, and rest
    d_1pl = np.delete(d_1pl, idx)
    p_r = np.delete(p_r, idx)
    rest = np.delete(rest, idx)
    # warn user about values with abs()>10
    if idx.size > 0:
        print("Warning: There are values in the difficulty array with abs()>10. These values have been removed from the plot.")

    r2 = scipy.stats.pearsonr(d_1pl.flatten(),p_r)[0]
    p =  scipy.stats.pearsonr(d_1pl.flatten(),p_r)[1]
    plt.figure(figsize=(8.5,5))
    plt.scatter(d_1pl, p_r, alpha=0.4, color='black')   
    
    #if covid_20.size > 0:
    #    plt.scatter(-d_1pl[covid_20], p_r[covid_20], alpha=0.6, color='tab:red', label='pandemic course offerings (2020-2022)')
    plt.xlabel('difficulty')
    plt.ylabel('pass rate')
    if p<0.001:
        plt.title("Pearson r = " + str("%.3f" % r2) + ", p < " + str("%.3f" % 0.001))
    else:
        plt.title("Pearson r = " + str("%.3f" % r2) + ", p = " + str("%.3f" % p))
    plt.savefig(data_folder + 'plots/validity_diff.pdf', format="pdf")#,  bbox_inches = "tight")
    #plt.show()

    plt.figure(figsize=(8.5,5))
    r2 = scipy.stats.pearsonr(abi_1pl.flatten(), gpa_copy)[0]
    p =  scipy.stats.pearsonr(abi_1pl.flatten(),gpa_copy)[1]
    plt.scatter(abi_1pl, gpa_copy, alpha=0.5, color='black')
    plt.xlabel('student trait')
    plt.ylabel('gpa')
    if p<0.001:
        plt.title("Pearson r = " + str("%.3f" % r2) + ", p < " + str("%.3f" % 0.001))
    else:
        plt.title("Pearson r = " + str("%.3f" % r2) + ", p = " + str("%.3f" % p))
    plt.savefig(data_folder + 'plots/validity_abi.pdf', format="pdf")
   

def regression_validation(degree, model, verbose=False, save_path=''):
    data_folder = parent_dir + '/data/real/' + degree + '/'
    temp_model = ''
    # translate model name:
    if "mmod1" in model:
        temp_model = "1pl"
    else:
        temp_model = "2pl"
    
    if "dim1" in model:
        temp_model += "_1dim"
        dim = 1
    elif "dim2" in model:
        temp_model += "_2dim"
        dim = 2
    elif "dim3" in model:
        temp_model += "_3dim"
        dim = 3
    else:
        temp_model += "_1dim"
        dim=1

    # Load the data
    tagged_input = np.array(pd.read_csv(data_folder + "binary_reduced/taggedRInput.csv", header=None))
    
    items = np.array(pd.read_csv(data_folder + "binary_reduced/item_ids.csv", header=None)).flatten()
    
    p_r = np.array(pd.read_csv(data_folder +  "binary_reduced/pass_rates.csv", header=None))[1:,1].astype(float)
    
    
    d_1pl = np.array(pd.read_csv(data_folder + "irt_results/"+ temp_model +"/diff.csv"))
    
    gpa = np.array(pd.read_csv(data_folder + "non_binary/gpas.csv", header=None))
    ids_non_bin = gpa[1:,0]
    gpa = gpa[1:,1].astype(float)
    
    gpa_bin = np.array(pd.read_csv(data_folder + "binary_reduced/gpas.csv", header=None))
    ids_binary = gpa_bin[1:,0]

    abi_1pl = np.array(pd.read_csv(data_folder + "irt_results/"+ temp_model +"/abilities.csv"))
    pred = np.array(pd.read_csv(data_folder + "irt_results/"+ temp_model +"/pred.csv"))
    pred = pred.T
    if dim == 1:
        rows = [
            {
                'abi': abi_1pl[student_index],
                'dif': d_1pl[course_index],
                'gpa': gpa[student_index],
                'pr': p_r[course_index],
                'label': tagged_input[course_index, student_index],
                'pred': pred[course_index, student_index]
            }
            for student_index in range(tagged_input.shape[1])
            for course_index in range(tagged_input.shape[0])
            if tagged_input[course_index, student_index] != MISSING_VALUE
        ]
        df = pd.DataFrame(rows)
        X_1 = df.drop(columns=['label', 'gpa', 'pr', 'pred'])
        X_2 = df.drop(columns=['label', 'abi', 'dif', 'pred'])
        y = df['label'].values
    elif dim == 2:
        rows = [
            {
                'abi_0': abi_1pl[student_index, 0],
                'abi_1': abi_1pl[student_index, 1],
                'dif_0': d_1pl[course_index, 0],
                'dif_1': d_1pl[course_index, 1],
                'gpa': gpa[student_index],
                'pr': p_r[course_index],
                'label': tagged_input[course_index, student_index],
                'pred': pred[course_index, student_index]
            }
            for student_index in range(tagged_input.shape[1])
            for course_index in range(tagged_input.shape[0])
            if tagged_input[course_index, student_index] != MISSING_VALUE
        ]
        df = pd.DataFrame(rows)
        # delete rows with nan values
        df = df.dropna()
        # warn user about nan values
        if df.shape[0] < tagged_input.shape[0]*tagged_input.shape[1]:
            print("Warning: There are nan values in the dataframe. These values have been removed from the dataframe.")
        
        # delete rows with +inf or -inf values
        df = df[(df != np.inf).all(1)]
        df = df[(df != -np.inf).all(1)]
        # warn user about inf values
        if df.shape[0] < tagged_input.shape[0]*tagged_input.shape[1]:
            print("Warning: There are inf values in the dataframe. These values have been removed from the dataframe.")

        # reset index
        df = df.reset_index(drop=True)

        X_1 = df.drop(columns=['label', 'gpa', 'pr', 'pred'])
        X_2 = df.drop(columns=['label', 'abi_0', 'dif_0', 'abi_1', 'dif_1', 'pred'])
        y = df['label'].values
    elif dim == 3:
        rows = [
            {
                'abi_0': abi_1pl[student_index, 0],
                'abi_1': abi_1pl[student_index, 1],
                'abi_2': abi_1pl[student_index, 2],
                'dif_0': d_1pl[course_index, 0],
                'dif_1': d_1pl[course_index, 1],
                'dif_2': d_1pl[course_index, 2],
                'gpa': gpa[student_index],
                'pr': p_r[course_index],
                'label': tagged_input[course_index, student_index],
                'pred': pred[course_index, student_index]
            }
            for student_index in range(tagged_input.shape[1])
            for course_index in range(tagged_input.shape[0])
            if tagged_input[course_index, student_index] != MISSING_VALUE 
        ]
        df = pd.DataFrame(rows)
        # delete rows with nan values
        df = df.dropna()
        # warn user about nan values
        if df.shape[0] < tagged_input.shape[0]*tagged_input.shape[1]:
            print("Warning: There are nan values in the dataframe. These values have been removed from the dataframe.")
        
        # delete rows with +inf or -inf values
        df = df[(df != np.inf).all(1)]
        df = df[(df != -np.inf).all(1)]
        # warn user about inf values
        if df.shape[0] < tagged_input.shape[0]*tagged_input.shape[1]:
            print("Warning: There are inf values in the dataframe. These values have been removed from the dataframe.")

        # reset index
        df = df.reset_index(drop=True)
        X_1 = df.drop(columns=['label', 'gpa', 'pr', 'pred'])
        X_2 = df.drop(columns=['label', 'abi_0', 'dif_0', 'abi_1', 'dif_1', 'abi_2', 'dif_2', 'pred'])
        y = df['label'].values



    # fit logistic regression only in sample
    from sklearn.linear_model import LogisticRegression

    clf_1 = LogisticRegression(random_state=0).fit(X_1, y)
    clf_2 = LogisticRegression(random_state=0).fit(X_2, y)

    # print ACC, AUC, NLL, RMSE for both models in sample in a table:
    from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, mean_squared_error
    #print()
    y_pred_1 = np.round(df['pred'].values)
    y_pred_2 = clf_2.predict(X_2)
    
    y_pred_1_prob = df['pred'].values
    #clf_1.predict_proba(X_1)[:,1]
    y_pred_2_prob = clf_2.predict_proba(X_2)[:,1]
    
    # y_pred_2_prob = X_1['']


    acc_1 = accuracy_score(y, y_pred_1)
    acc_2 = accuracy_score(y, y_pred_2)
    auc_1 = roc_auc_score(y, y_pred_1_prob)
    auc_2 = roc_auc_score(y, y_pred_2_prob)
    nll_1 = log_loss(y, y_pred_1_prob)
    nll_2 = log_loss(y, y_pred_2_prob)
    rmse_1 = mean_squared_error(y, y_pred_1_prob)
    rmse_2 = mean_squared_error(y, y_pred_2_prob)

    # print results as dataframe
    results = pd.DataFrame(
        [
            [acc_1, auc_1, nll_1, rmse_1],
            [acc_2, auc_2, nll_2, rmse_2]
        ],
        columns=['ACC', 'AUC', 'NLL', 'RMSE'],
        index=[model, 'gpa + pr']
    )  
    print(results)
    
    return


if __name__ == '__main__':
    regression_validation('CompSci', 'mmod2_dim3', verbose=False, save_path='')