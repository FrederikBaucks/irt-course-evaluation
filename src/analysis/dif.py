# DIF analysis for IRT items 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
from scipy import stats
import os
import sys
import statsmodels.api as sm
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Add the parent directory to sys.path
sys.path.append(parent_dir)
sys.path.append(parent_dir + '/src')
MISSING_VALUE = -99999
import Classes

from statsmodels.formula.api import glm


import scipy.stats as stats
def data_tests(ability, group_1_ids, group_2_ids):
    # check if the two groups are not empty
    if group_1_ids.empty or group_2_ids.empty:
        raise ValueError('One of the groups is empty')
    
    # check if the two groups are too small     
    if len(group_1_ids) < 10 or len(group_2_ids) < 10:
        raise ValueError('One of the groups is too small (<10)')

    # check if there are no duplicates
    if ability['student_id'].duplicated().any():
        raise ValueError('There are duplicates in the ability data')
    
    # check if the two group ids are disjoint
    if not set(group_1_ids['student_id']).isdisjoint(set(group_2_ids['student_id'])):
        raise ValueError('The two groups are not disjoint')

def plot_data(combined_data):
    # Adjust the figsize and layout as needed
    fig = plt.figure(figsize=(15, 10))

    group_1 = combined_data[combined_data['group'] == 'group_1']
    group_2 = combined_data[combined_data['group'] == 'group_2']

    # Scatter Plots for each pair of ability dimensions
    pairs = [(1, 2), (1, 3), (2, 3)]
    for i, (dim1, dim2) in enumerate(pairs, 1):
        ax_scatter = fig.add_subplot(4, 3, i)
        sns.scatterplot(x=f'ability_{dim1}', y=f'ability_{dim2}', hue='group', data=group_1, ax=ax_scatter, palette='Blues', legend=False)
        sns.scatterplot(x=f'ability_{dim1}', y=f'ability_{dim2}', hue='group', data=group_2, ax=ax_scatter, palette='Reds', legend=True if i == 1 else False)
        ax_scatter.set_title(f'Scatter Plot of Ability {dim1} vs. Ability {dim2}')
        ax_scatter.set_xlabel(f'Ability {dim1}')
        ax_scatter.set_ylabel(f'Ability {dim2}')

    # Histograms and Box Plots for each ability dimension
    for i in range(1, 4):
        # Histogram
        ax_hist = fig.add_subplot(4, 3, 3 + i)
        sns.histplot(group_1[f'ability_{i}'], color='blue', label='Group 1', kde=True, ax=ax_hist)
        sns.histplot(group_2[f'ability_{i}'], color='red', label='Group 2', kde=True, ax=ax_hist)
        ax_hist.set_title(f'Distribution of Ability {i}')
        ax_hist.set_xlabel(f'Ability {i} Score')
        ax_hist.set_ylabel('Frequency')
        ax_hist.legend()

        # Box Plot
        ax_box = fig.add_subplot(4, 3, 6 + i)
        sns.boxplot(data=combined_data, x='group', y=f'ability_{i}', ax=ax_box)
        ax_box.set_title(f'Ability {i} Scores by Group')
        ax_box.set_xlabel('Group')
        ax_box.set_ylabel(f'Ability {i} Score')

    plt.tight_layout()
    plt.show()

def stud_ids_by_course(degree_1, degree_2, course_id):
    import pickle
    degree = degree_1 + '_' + degree_2
    data_folder = parent_dir + '/data/real/' + degree + '/'
    anova = pd.read_csv(data_folder + '/irt_results/model_selection/anova_1PL_2PL_1DIM.csv')
    # get index of best model aka lowest BIC
    best_model = anova['BIC'].idxmin()
    #print(anova)
    # translate model_name
    trans = {   'mmod1': '1pl_1dim', 
        'mmod2_dim1': '2pl_1dim',
        'mmod2_dim2': '2pl_2dim',
        'mmod2_dim3': '2pl_3dim'}
    best_model = trans[best_model]

    data_folder_1 = parent_dir + '/data/real/' + degree_1 + '/'
    studs_ids_d1 = pd.read_csv(data_folder_1 + 
                           'binary_reduced/student_ids.csv', header=None)
    
    data_folder_2 = parent_dir + '/data/real/' + degree_2 + '/'
    studs_ids_d2 = pd.read_csv(data_folder_2 + 
                           'binary_reduced/student_ids.csv', header=None)  
    with open(data_folder+'StudentGroup_filtered.pickle', 'rb') as f:
        stud_group = pickle.load(f)



    studs_1 = []
    studs_2 = []
    for student in stud_group.students:
    
        if course_id in student.courseNames:
            if student.name in studs_ids_d1.values:
                studs_1.append(student.name)
            elif student.name in studs_ids_d2.values:
                studs_2.append(student.name)
            else:
                continue
    if pd.DataFrame(studs_1).shape == (0,0) or pd.DataFrame(studs_2).shape == (0,0):
        raise ValueError('No students found for one of the groups. Please check your input.')
    return pd.DataFrame(studs_1), pd.DataFrame(studs_2)

def compare_abilities(degree, group_1_ids, group_2_ids, plot=False):
    # Load data
    data_folder = parent_dir + '/data/real/' + degree + '/'

    anova = pd.read_csv(data_folder + '/irt_results/model_selection/anova_1PL_2PL_1DIM.csv')
    # get index of best model aka lowest BIC
    best_model = anova['BIC'].idxmin()
    best_model = 'mmod1'
    #print(anova)
    # translate model_name
    trans = {   'mmod1': '1pl_1dim', 
        'mmod2_dim1': '2pl_1dim',
        'mmod2_dim2': '2pl_2dim',
        'mmod2_dim3': '2pl_3dim'}


    # read ability estimates
    ability = pd.read_csv(data_folder + 
                          '/irt_results/'+ trans[best_model] +'/abilities.csv')
    

    # reset index
    ability = ability.reset_index(drop=True)

    # check dimension of ability estimates
    if ability.shape[1] < 3:
        # drop column names
        ability.columns = range(ability.shape[1]) 
        # add columns
        ability = ability.reindex(columns=range(3), fill_value=0)

    # read student ids 
    stud_ids = pd.read_csv(data_folder + 
                           'binary_reduced/student_ids.csv', header=None)
    # merge student ids and ability estimates
    ability = pd.concat([stud_ids, ability], axis=1)
    # set column names
    ability.columns = ['student_id', 'ability_1', 'ability_2', 'ability_3']

    # test the data for DIF
    data_tests(ability, group_1_ids, group_2_ids)



    # get the mean and std of the ability estimates of the two groups
    group_1 = ability[ability['student_id'].isin(group_1_ids['student_id'])]
    group_2 = ability[ability['student_id'].isin(group_2_ids['student_id'])]

    # ignore SettingWithCopyWarning
    pd.options.mode.chained_assignment = None  # default='warn'

    # Add a 'group' column to each DataFrame
    group_1['group'] = 'group_1'
    group_2['group'] = 'group_2'

    # Combine the DataFrames
    combined_data = pd.concat([group_1, group_2])

    # del group_1 and group_2 from memory
    del group_1
    del group_2
    del ability

    if plot:
        plot_data(combined_data)
    
    # get the mean and std of the ability estimates of the two groups
    group_1_mean = combined_data[combined_data['group'] == 'group_1'].mean(numeric_only=True)
    group_2_mean = combined_data[combined_data['group'] == 'group_2'].mean(numeric_only=True)
    group_1_std = combined_data[combined_data['group'] == 'group_1'].std(numeric_only=True)
    group_2_std = combined_data[combined_data['group'] == 'group_2'].std(numeric_only=True)

    # get the difference of the means
    mean_diff = np.array(group_1_mean) - np.array(group_2_mean )
    
    # get the difference of the stds
    std_diff = group_1_std - group_2_std
    # get the t-value
    t_value = mean_diff / np.sqrt((group_1_std**2 / len(group_1_ids)) + (group_2_std**2 / len(group_2_ids)))
    # get the p-value
    p_value = 2 * (1 - stats.t.cdf(abs(t_value), len(combined_data) - 2))
    if plot:
        # print results in a table format
        print('Ability Analysis Results')
        print('---------------------')
        print(f'Group 1 Size: {len(group_1_ids)}')
        print(f'Group 2 Size: {len(group_2_ids)}')
        print('---------------------')
        print('Mean Differences')
        print('---------------------')
        print(f'Ability 1: {mean_diff[0]}')
        print(f'Ability 2: {mean_diff[1]}')
        print(f'Ability 3: {mean_diff[2]}')
        print('---------------------')
        print('Std Differences')
        print('---------------------')
        print(f'Ability 1: {std_diff[0]}')
        print(f'Ability 2: {std_diff[1]}')
        print(f'Ability 3: {std_diff[2]}')
        print('---------------------')
        print('T-Test Results')
        print('---------------------')
        print(f'Ability 1: t-value = {t_value[0]}, p-value = {p_value[0]}')
        print(f'Ability 2: t-value = {t_value[1]}, p-value = {p_value[1]}')
        print(f'Ability 3: t-value = {t_value[2]}, p-value = {p_value[2]}')
        print('---------------------')
    # save results in DataFrame
    results =  pd.DataFrame()
    results['mean_g1'] = group_1_mean
    results['mean_g2'] = group_2_mean
    results['std_g1'] = group_1_std
    results['std_g2'] = group_2_std
    results['mean_dif'] = mean_diff
    results['std_dif'] = std_diff
    results['t'] = t_value
    results['p'] = p_value
    return results

def ability(data_folder):

    anova = pd.read_csv(data_folder + '/irt_results/model_selection/anova_1PL_2PL_1DIM.csv')
    # get index of best model aka lowest BIC
    best_model = anova['BIC'].idxmin()
    #print(anova)
    # translate model_name
    trans = {   'mmod1': '1pl_1dim', 
        'mmod2_dim1': '2pl_1dim',
        'mmod2_dim2': '2pl_2dim',
        'mmod2_dim3': '2pl_3dim'}


    # read ability estimates
    ability = pd.read_csv(data_folder + 
                          '/irt_results/'+ trans[best_model] +'/abilities.csv')
    

    # reset index
    ability = ability.reset_index(drop=True)

    # check dimension of ability estimates
    if ability.shape[1] < 3:
        # drop column names
        ability.columns = range(ability.shape[1]) 
        # add columns
        ability = ability.reindex(columns=range(3), fill_value=0)

    # read student ids 
    stud_ids = pd.read_csv(data_folder + 
                           'binary_reduced/student_ids.csv', header=None)
    # merge student ids and ability estimates
    ability = pd.concat([stud_ids, ability], axis=1)
    # set column names
    ability.columns = ['student_id', 'ability_1', 'ability_2', 'ability_3']
    return ability


def diff(data_folder):

    anova = pd.read_csv(data_folder + '/irt_results/model_selection/anova_1PL_2PL_1DIM.csv')
    # get index of best model aka lowest BIC
    best_model = anova['BIC'].idxmin()
    #print(anova)
    # translate model_name
    trans = {   'mmod1': '1pl_1dim', 
        'mmod2_dim1': '2pl_1dim',
        'mmod2_dim2': '2pl_2dim',
        'mmod2_dim3': '2pl_3dim'}


    # read ability estimates
    difficulty = pd.read_csv(data_folder + 
                          '/irt_results/'+ trans[best_model] +'/diff.csv')
    

    # reset index
    difficulty = difficulty.reset_index(drop=True)

    # check dimension of ability estimates
    if difficulty.shape[1] < 3:
        # drop column names
        difficulty.columns = range(difficulty.shape[1]) 
        # add columns
        difficulty = difficulty.reindex(columns=range(3), fill_value=0)

    # read student ids 
    item_ids = pd.read_csv(data_folder + 
                           'binary_reduced/item_ids.csv', header=None)
    # merge student ids and difficulty estimates
    difficulty = pd.concat([item_ids, difficulty], axis=1)
    # set column names
    difficulty.columns = ['item_id', 'difficulty_1', 'difficulty_2', 'difficulty_3']
    return difficulty, best_model


def standardize(series):
    return (series - series.mean()) / series.std()


def log_dif_analysis(degree_1, degree_2, group_1_ids, group_2_ids, plot=False):
    # Load data
    if degree_2 is None:
        degree = degree_1
    else:
        degree = degree_1 + '_' + degree_2
    data_folder = parent_dir + '/data/real/' + degree + '/'

    abilities = ability(data_folder)
    difficulty, best_model = diff(data_folder)

    
    responses = pd.read_csv(data_folder + 
                        'binary_reduced/taggedRInput.csv', header=None)
    stud_ids = pd.read_csv(data_folder + 
                        'binary_reduced/student_ids.csv', header=None)
    course_ids = pd.read_csv(data_folder + 
                        'binary_reduced/item_ids.csv', header=None)
    
    # structure data set:

    mh_test_data = pd.DataFrame(columns=['student', 'item_response', 'group'])
    rows_to_add = []
    for s_iter, s_id in enumerate(stud_ids.values):
        #sum(responses.iloc[:, s_iter][responses.iloc[:, s_iter]!=MISSING_VALUE])
        if s_id[0] in group_1_ids.values:
            group = -1
        elif s_id[0] in group_2_ids.values:
            group = 1   
        else:
            continue
        for c_iter, c_id in enumerate(course_ids.values):
            # get value
            value = responses.iloc[c_iter,s_iter]
            if value in [0,1]:
                # concat to data frame not append
                row = {'student': s_id[0], 'item': c_id[0], 'item_response': value, 'group': float(group)}
                rows_to_add.append(row)
    new_rows_df = pd.DataFrame(rows_to_add)
    try:
        mh_test_data = pd.concat([mh_test_data, new_rows_df], ignore_index=True)
    except:
        print('concat of mh_test_data, new_rows_df failed')
    #print('flag 1')

    print('merge student abilities with mh_test_data')
    # merge abilities with mh_test_data such that abilities are added in each row corresponing to the student
    mh_test_data = pd.merge(mh_test_data, abilities, how='left', left_on='student', right_on='student_id')
    print('merge difficulty with mh_test_data')
    # merge difficulty with mh_test_data such that difficulty is added in each row corresponing to the item
    mh_test_data = pd.merge(mh_test_data, difficulty, how='left', left_on='item', right_on='item_id')
    #ok print(mh_test_data.head())

    #print('flag 2')

    #sort after item name
    print('sort values mh_test_data and drop nan values')
    mh_test_data = mh_test_data.sort_values(by=['item'])
    mh_test_data = mh_test_data.dropna()

    #ok print(mh_test_data.head())

    #print(best_model)
    # get dim
    if 'dim1' in best_model:
        dim = 1
    elif 'dim2' in best_model:
        dim = 2
    elif 'dim3' in best_model:
        dim = 3
    else:
        dim = 1

    # Assuming 'mh_test_data' is your DataFrame

    # Get a list of unique items
    unique_items = mh_test_data['item'].unique()
    best_model = 'mmod1'

    #print(best_model)
    # for now we assume only use the the Rasch model for DIF analysis, therefore we delete abilties and difficulties
    print('drop abilities and difficulties')
    mh_test_data = mh_test_data.drop(['ability_2', 'ability_3', 'difficulty_2', 'difficulty_3'], axis=1)

    # Instead of using anchors in the modelling of student abilities, we will transform the 
    # abilities group-wise to have zero mean an unit variance. This will allow us to find DIF courses
    # a posteriori.

    # step 1 - transform the abilities to have zero mean and unit variance
    #print(mh_test_data.head())
    #print('flag3',  mh_test_data.groupby('group')['ability_1'].apply(standardize).shape, mh_test_data.shape)
    # print('scale abilities')
    # for name, group in mh_test_data.groupby('group'):
    #     if len(group) <= 1 or group['ability_1'].std() == 0:
    #         print(f"Issue in len or std group: {name}", len(group), group['ability_1'].std())
    #     standardized = standardize(group['ability_1'])
    #     # Check for issues like NaNs or Infs in 'standardized'
    #     if standardized.isna().any() or np.isinf(standardized).any():
    #         print(f"Issue in standardization of group {name}", standardized.isna().any(), np.isinf(standardized).any())    
    # mh_test_data.reset_index(drop=True, inplace=True)
    # scaled_ability_1 = mh_test_data.groupby('group')['ability_1'].apply(standardize)
    # print('scaled_ability_1', scaled_ability_1.shape)
    # print('mh_test_data', mh_test_data.shape)
    # scaled_ability_1.reset_index(drop=True, inplace=True)
    # mh_test_data['scaled_ability_1'] = scaled_ability_1 

#mh_test_data.groupby('group')['ability_1'].apply(standardize)
    
    #print(mh_test_data['scaled_ability_1'])
    #print('flag4')
    # step 2 - fit the model with the transformed abilities
    


    # step 3 - check for DIF

    # # Preliminary DIF analysis for potential anchor item identification
    # potential_anchors = []
    # for item in unique_items:
    #     item_data = mh_test_data[mh_test_data['item'] == item]
    #     # Replace all ' ' in item names with '_'
    #     item_data.columns = [col.replace(' ', '_') for col in item_data.columns]
       
       
    #     #print(item_data.head())
    #     formula = 'item_response ~ group + ability_1 + ability_2 + ability_3'  # total_score as a proxy for ability
    #     model = glm(formula, data=item_data, family=sm.families.Binomial()).fit(disp=0)
    #     if model.pvalues['group']>0.05/len(unique_items):
    #         potential_anchors.append(item)
    # potential_anchors = [anchor.replace(' ', '_') for anchor in potential_anchors]
    # print(f'Number of anchor items: {len(potential_anchors)}')
    # print(potential_anchors)

    # # add anchor item responses to the mh_test_data as colums with the item name
    # for anchor in potential_anchors:
    #     mh_test_data[anchor] = mh_test_data['item_response'][mh_test_data['item'] == anchor]
    
    
    mean_ability = np.mean(mh_test_data['ability_1'])
    print(mean_ability.shape)
    # Dictionary to store logistic regression results for each item
    logistic_results = {}
    group_sizes = []
    #dif_items = [item for item in unique_items if item not in potential_anchors]
    print('fit LR DIF models for each item')
    for item in unique_items:

        item_data = mh_test_data[mh_test_data['item'] == item]
        if item_data.groupby('group')['ability_1'].mean().shape==(2,):
            print(
                    item,  '(1 vs -1)', 
                    'ability ', 
                    round(item_data.groupby('group')['ability_1'].mean()[1] - item_data.groupby('group')['ability_1'].mean()[-1], 3),
                    'pr ',
                    round(item_data.groupby('group')['item_response'].mean()[1]- item_data.groupby('group')['item_response'].mean()[-1], 3)
                )
        
        # Replace all ' ' in item names with '_'
        item_data.columns = [col.replace(' ', '_') for col in item_data.columns]

        item_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        item_data.fillna(0, inplace=True)

        # find mean item_response for each group
        mean_item_response = item_data.groupby('group')['item_response'].mean()

        if mean_item_response.shape[0] < 2:
            groups_in_item = False
            pr_diff = 0
            var_diff = 0
            pr_diff_ttest_p_val = 1
            pr_diff_err = 0
        else:
            groups_in_item = True
            pr_diff = mean_item_response.iloc[1]-mean_item_response.iloc[0]
            g_neg1 = item_data[item_data["group"] == -1]
            g_neg1_scores = g_neg1["item_response"].values
            g_pos1 = item_data[item_data["group"] == 1]
            g_pos1_scores = g_pos1["item_response"].values
            pr_diff_ttest_p_val = stats.ttest_ind(g_neg1_scores, g_pos1_scores)[1]
            pr_diff_err = 1.96*(np.sqrt(np.std(g_neg1_scores)**2/len(g_neg1_scores) + np.std(g_pos1_scores)**2/len(g_pos1_scores)))
            
        # Updated formula with anchor items
        formula_1 = 'item_response ~ group + 1'
        formula_2 = 'item_response ~ 1'

        offset_data_rasch = item_data['ability_1']
        offset_data = item_data['ability_1'] - item_data['difficulty_1'] 

        baseline_model = model = glm(formula_2, data=item_data, family=sm.families.Binomial(), offset=offset_data_rasch)
        baseline_result = baseline_model.fit()
        model = glm(formula_1, data=item_data, family=sm.families.Binomial(), offset=offset_data)
        result = model.fit()
        

        
        item_diff = item_data['difficulty_1'].mean()
        group_dif = result.params['group']
        err = result.bse
        interecept = result.params['Intercept']

        if mean_item_response.shape[0] < 2:
            log_pr_dif = 0
            log_pr_err = 0
        else:
            
            log_group_pos1 = 1/(1+np.exp(-(mean_ability - item_diff + group_dif + interecept)))
            log_group_neg1 = 1/(1+np.exp(-(mean_ability - item_diff + interecept - group_dif)))
            log_pr_dif = log_group_pos1 - log_group_neg1

            log_pr_err = abs(log_group_pos1 - 1/(1+np.exp(-(mean_ability - item_diff + group_dif + 1.95 * err['group'] + interecept))))
            #print(log_pr_err)
            #print(log_pr_dif)
        

    
        #print(log_pr_dif, result.params['group'])
  
        


        # Likelihood Ratio Test
        # Calculate the test statistic
        lr_statistic = -2 * (baseline_result.llf - result.llf)

        # Degrees of freedom (number of extra parameters in Model 2)
        df = len(result.params) - len(baseline_result.params)
       
        # Calculate the p-value
        p_value = stats.chi2.sf(lr_statistic, df)
        logistic_results[item] = [log_pr_dif, result.pvalues, log_pr_err, p_value, pr_diff, pr_diff_ttest_p_val, pr_diff_err]
        
        if p_value < 0.05:
            item_data['predicted_probability'] = result.predict(item_data, offset=offset_data)
            mean_probs = item_data.groupby(['item_id', 'group'])['predicted_probability'].mean().reset_index()
            
            mean_diff_per_course = mean_probs.pivot(index='item_id', columns='group', values='predicted_probability')
            #print(mean_diff_per_course)
            mean_diff_per_course['mean_diff'] = mean_diff_per_course[-1] - mean_diff_per_course[1]
        # append group sizes of each item
        group_size_dict = item_data['group'].value_counts().to_dict()
        # set key to item name
        group_size_dict['item'] = item
        group_size_dict['beta_1'] = result.params['group']
        group_sizes.append(group_size_dict)
        # # Likelihood Ratio Test
        # # Calculate the test statistic
        # lr_statistic = -2 * (baseline_result.llf - result.llf)

        # # Degrees of freedom (number of extra parameters in Model 2)
        # df = len(result.params) - len(baseline_result.params)
       
        # # Calculate the p-value
        # p_value = stats.chi2.sf(lr_statistic, df)
        
        # print(item)
        # print(f"Likelihood Ratio Statistic: {lr_statistic}, p-value: {p_value}, degrees of freedom: {df}")

        # # Check log-likelihoods
        # print("Log-likelihood of Model 1:", result.llf)
        # print("Log-likelihood of Model 2:", baseline_result.llf)

        # # Compare AIC and BIC
        # aic1, bic1 = result.aic, result.bic
        # aic2, bic2 = baseline_result.aic, baseline_result.bic
        # print(f"model - AIC: {aic1}, BIC: {bic1}")
        # print(f"baseline - AIC: {aic2}, BIC: {bic2}")
            
        
    # print table with item names and group sizes
    # Set the maximum number of rows and columns to display
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 20)
    group_sizes = pd.DataFrame(group_sizes)
    print(group_sizes)
    #print('group_sizes', group_sizes)
    #print('flag5', logistic_results) 
    # for item in unique_items:
    #     # Filter data for the current item
    #     item_data = mh_test_data[mh_test_data['item'] == item]
    #     #print(item_data.isna().sum())
    #     #print(item_data.isin([np.inf, -np.inf]).sum())
    #     item_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    #     item_data.fillna(0, inplace=True)
    #     # Define your model
    #     #X = item_data[['total_score', 'group']]  # Independent variables
    #     #y = item_data['item_response']  # Dependent variable



    #     # Logistic regression model
    #     formula = 'item_response ~ difficulty_1 + difficulty_2 + difficulty_3 + group'
    #     for i in range(1, dim+1):
    #         formula += f' + ability_{i}:group + ability_{i}'

    #     model = glm(formula, data=item_data, family=sm.families.Binomial())

    #     # Fit the model
    #     result = model.fit()


    #     # # Add a constant to the model (intercept)
    #     # X = sm.add_constant(X)

    #     # # Fit the logistic regression model
    #     # model = sm.Logit(y, X).fit(disp=0)

    #     # Store the summary of the model
    #     logistic_results[item] = [result.params, result.pvalues, result.bse]

    # go over each item but the anchor items 
    # Replace all ' ' in item names with '_'
    
    # dif_items = [item for item in unique_items if item not in potential_anchors]
    # for item in dif_items:

    #     item_data = mh_test_data[mh_test_data['item'] == item]
    #     # Replace all ' ' in item names with '_'
    #     item_data.columns = [col.replace(' ', '_') for col in item_data.columns]

    #     item_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    #     item_data.fillna(0, inplace=True)

    #     # Include anchor items in the formula
    #     anchor_formula = ' + '.join([f'{anchor}:group + {anchor}' for anchor in potential_anchors])
        
    #     # Updated formula with anchor items
    #     formula = f'item_response ~ difficulty_1 + difficulty_2 + difficulty_3 + group + {anchor_formula}'
    #     for i in range(1, dim+1):
    #         formula += f' + ability_{i}:group + ability_{i}'

    #     model = glm(formula, data=item_data, family=sm.families.Binomial())
    #     result = model.fit()
    #     logistic_results[item] = [result.params, result.pvalues, result.bse]

    # Creating a DataFrame
    keys_to_keep = ['group']

    # Creating a DataFrame
    df_list = []
    print('build result df for each item')
    for course, (coeffs, pvals, bse, lr_pvals, pr_diff, pr_diff_ttest_p_val, pr_diff_err) in logistic_results.items():
        filtered_coeffs = coeffs
        filtered_pvals = pvals.filter(items=keys_to_keep)
        filtered_bse = bse
        filtered_pr_diff = pr_diff
        filtered_lr_pvals = lr_pvals
        temp_df = pd.DataFrame({
            'Course': course,
            'Parameter': filtered_pvals.index,
            'Coefficient': filtered_coeffs,
            'P-Value': filtered_pvals.values,
            'Standard Error': filtered_bse,
            'LR-test-P-Value': filtered_lr_pvals,
            'Pr(Diff)': filtered_pr_diff,
            'Pr(Diff) t-test P-Value': pr_diff_ttest_p_val,
            'Pr(Diff) Standard Error': pr_diff_err
        })
        df_list.append(temp_df)

    concise_df = pd.concat(df_list, ignore_index=True)
    dim = 1
    # We will use the Benjamini-Hochberg procedure to control the False Discovery Rate (FDR) seperately for each test.

    # Implementing Benjamini-Hochberg Procedure
    alpha = 0.05


    
    p_values = concise_df['P-Value'].values
    m = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p_values = p_values[sorted_indices]
    tt_test_p_values = concise_df['Pr(Diff) t-test P-Value'].values
    sorted_ttest_p_values = tt_test_p_values[sorted_indices]
    #print(sorted_ttest_p_values)
    ranks = np.arange(1, m + 1)
    # Calculate BH critical values
    bh_critical_values = (ranks / m) * alpha
    dif_results = concise_df.copy()
    # Find the largest p-value where p-value < BH critical value
    significant = sorted_p_values <= bh_critical_values
    tt_test_significant = sorted_ttest_p_values <= bh_critical_values
    # check if there is a value True in significant
    if True not in significant:
        significant = np.array([False])
        dif_results['color'] = 'lightgray'
        dif_results['ttest_color'] = 'lightgray'
    else:
        last_true = np.max(np.where(significant))
        ttest_last_true = np.max(np.where(tt_test_significant))
        # Get the threshold p-value
        bh_threshold = sorted_p_values[last_true]
        ttest_bh_threshold = sorted_ttest_p_values[ttest_last_true]
        print('Benjamini-Hochberg threshold: ', bh_threshold, ttest_bh_threshold)
        # Assign colors based on BH significance
        dif_results['color'] = np.where(dif_results['P-Value'] <= bh_threshold, 'green', 'green')
        dif_results['ttest_color'] = np.where(dif_results['Pr(Diff) t-test P-Value'] <= ttest_bh_threshold, 'pink', 'lightgray')

    #print(dif_results['ttest_color'])


    p_values = concise_df['LR-test-P-Value'].values
    m = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p_values = p_values[sorted_indices]
    sorted_ttest_p_values = tt_test_p_values[sorted_indices]
    ranks = np.arange(1, m + 1)
    # Calculate BH critical values
    bh_critical_values = (ranks / m) * alpha
    # Find the largest p-value where p-value < BH critical value
    significant = sorted_p_values <= bh_critical_values
    if True not in significant:
        significant = np.array([False])
        dif_results['color'] = 'lightgray'
    else:
        #print(significant)
        last_true = np.max(np.where(significant))
        # Get the threshold p-value
        bh_threshold = sorted_p_values[last_true]
        # Assign colors based on BH significance
        # Change color to orange if LR-test-P-Value is greater than BH threshols for each value in dif_results['color']
        change_idx = np.where(dif_results['LR-test-P-Value'] > bh_threshold)
        
        dif_results['color'].iloc[change_idx] = 'lightgray'


    if plot:
        try:
            # set seaborn style
            sns.set_style('whitegrid')
            # create new colum with color names
            #dif_results['color'] = np.where(dif_results['P-Value'] < (0.05/dif_results.shape[0]), 'blue', 'orange')
            
        
            list_ability = []
            for p, param in  enumerate(dif_results['Parameter'].unique().tolist()):
                # Filter data for ability_1:group and ability_2:group
                list_ability.append(dif_results[dif_results['Parameter'] == 'group'])
                # list_ability.append(dif_results[dif_results['Parameter'] == f'ability_{p+1}:group'])
            #print(dif_results) 
            # Creating subplots
            
            fig, axes = plt.subplots(nrows=p+1, ncols=1, figsize=(20, 5))
            # if dim>1:
            #     for i, df in enumerate(list_ability):
            #         # sort df by values of Coefficient
            #         df = df.sort_values(by=['Coefficient'])

            #         # Plot ability_1:group
            #         df.plot(kind='bar', x='Course', y='Coefficient', yerr='Standard Error', ax=axes[i], color=df['color'], label=f'ability_{i+1}:group')
            #         axes[i].set_ylim([-2.5, 2.5])
            #     # say that orange is insignificant
            #     axes[0].legend(['blue = significant', 'orange = insignificant'])
            if dim == 1:
                for i, df in enumerate(list_ability):
                    #print(df)
                     # sort df by values of Coefficient
                    df = df.sort_values(by=['Coefficient'])
                    # Plot ability_1:group
                    df.plot(kind='bar', x='Course', y='Coefficient', yerr='Standard Error', ax=axes, color=df['color'], label='group')

                    # df.plot(kind='bar', x='Course', y='Coefficient', yerr='Standard Error', ax=axes, color=df['color'], label=f'ability_{i+1}:group')
                    axes.set_ylim([-.5, .5])
                    axes.set_ylabel('DIF Effect')

                    

                # say that orange is insignificant
                # Create custom legend handles
                significant_patch = mpatches.Patch(color='green', label='Significant DIF')
                insignificant_patch = mpatches.Patch(color='gray', label='Insignificant DIF')
                #plt.title('Significance of Group Effects Across Courses: $\\mathbf{Negative \\, Effect}$ ~ Course is $\\mathbf{Easier \\, for \\, ' + degree_1 + '}$ Students, $\\mathbf{Positive \\, Effect}$ ~ Course is $\\mathbf{Easier \\, for \\, ' + degree_2 + '}$ Students')
                plt.title('Significance of Group Effects Across Courses: $\\mathbf{Negative \\, Effect}$ ~ Course is $\\mathbf{Easier \\, for \\, ' + 'Group 1' + '}$ Students, $\\mathbf{Positive \\, Effect}$ ~ Course is $\\mathbf{Easier \\, for \\, ' + 'Group 2' + '}$ Students')
                # Add the custom legend to the plot
                plt.legend(handles=[significant_patch, insignificant_patch])
                #axes.legend(['blue = significant', 'orange = insignificant'])
            # Adjust layout
            plt.tight_layout()

            # Show plot
            plt.show()


            fig, axes = plt.subplots(nrows=p+1, ncols=1, figsize=(20, 5))
            
            if dim == 1:
                for i, df in enumerate(list_ability):
                    #print(df)
                     # sort df by values of Coefficient
                    df = df.sort_values(by=['Coefficient'])
                    df.plot(kind='bar', x='Course', y='Pr(Diff)', yerr='Pr(Diff) Standard Error', ax=axes, color=df['ttest_color'], label='group', alpha=0.8)
                    # df.plot(kind='bar', x='Course', y='Coefficient', yerr='Standard Error', ax=axes, color=df['color'], label=f'ability_{i+1}:group')
                    axes.set_ylim([-0.5, 0.5])
                    axes.set_ylabel('Pass Rate Difference')

                    

                # say that orange is insignificant
                # Create custom legend handles
                insignificant_patch = mpatches.Patch(color='gray', label='Insignificant DIF')
                pr_diff_patch = mpatches.Patch(color='pink', label='Difference in Pass Rates')
                #plt.title('Significance of Group Effects Across Courses: $\\mathbf{Negative \\, Effect}$ ~ Course is $\\mathbf{Easier \\, for \\, ' + degree_1 + '}$ Students, $\\mathbf{Positive \\, Effect}$ ~ Course is $\\mathbf{Easier \\, for \\, ' + degree_2 + '}$ Students')
                plt.title('Significance of Group Effects Across Courses: $\\mathbf{Negative \\, Effect}$ ~ Course is $\\mathbf{Easier \\, for \\, ' + 'Group 1' + '}$ Students, $\\mathbf{Positive \\, Effect}$ ~ Course is $\\mathbf{Easier \\, for \\, ' + 'Group 2' + '}$ Students')
                # Add the custom legend to the plot
                plt.legend(handles=[pr_diff_patch, insignificant_patch])
                #axes.legend(['blue = significant', 'orange = insignificant'])
            # Adjust layout
            plt.tight_layout()

            # Show plot
            plt.show()
        except Exception as e:
            print(e)
            print('DIF plotting failed')

    return concise_df
    
if __name__ == '__main__':
    degree = 'a_b'
    data_folder = parent_dir + '/data/real/' + degree + '/'
    stud_ids = pd.read_csv(data_folder + 
                           'binary_reduced/student_ids.csv', header=None)
    # choose two groups with no intersection
    group_1_ids = stud_ids.iloc[:200]
    group_2_ids = stud_ids.iloc[200:400]
    #item = '1'
    #group_1_ids = pd.DataFrame(['a_student_0', 'a_student_1'])
    group_1_ids.columns = ['student_id']
    #group_2_ids = pd.DataFrame(['b_student_0', 'b_student_1'])
    group_2_ids.columns = ['student_id']
    score = '1'
    compare_abilities(degree, group_1_ids, group_2_ids, plot=False)