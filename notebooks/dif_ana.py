import pandas as pd
import numpy as np
import os
import traceback


# import dif from src/analysis
import sys


def groups_by_degree(degree_1, degree_2):
    MISSING_VALUE = -99999
    parent_dir = os.path.dirname(os.getcwd())
    analysis_dir = parent_dir + '/src/analysis/'
    sys.path.append(analysis_dir)

    degree = degree_1 + '_' + degree_2

    data_folder = parent_dir + '/data/real/' + degree + '/'
    data_folder_1 = parent_dir + '/data/real/' + degree_1 + '/'
    data_folder_2 = parent_dir + '/data/real/' + degree_2 + '/'


    stud_ids = pd.read_csv(data_folder + 
                            'binary_reduced/student_ids.csv', header=None)
    stud_ids_1 = pd.read_csv(data_folder_1 + 
                            'binary_reduced/student_ids.csv', header=None)
    stud_ids_2 = pd.read_csv(data_folder_2 + 
                            'binary_reduced/student_ids.csv', header=None)
    
    # create data frame of stud_ids_1 that are also in stud_ids
    group_1_ids = pd.merge(stud_ids, stud_ids_1, how='inner', on=0)
    group_2_ids = pd.merge(stud_ids, stud_ids_2, how='inner', on=0)
    return group_1_ids, group_2_ids    

def random_groups_single_degree(degree_1):
    MISSING_VALUE = -99999
    parent_dir = os.path.dirname(os.getcwd())
    analysis_dir = parent_dir + '/src/analysis/'
    sys.path.append(analysis_dir)

    degree = degree_1

    data_folder = parent_dir + '/data/real/' + degree + '/'
    data_folder_1 = parent_dir + '/data/real/' + degree_1 + '/'


    stud_ids = pd.read_csv(data_folder + 
                            'binary_reduced/student_ids.csv', header=None)
    ids = pd.read_csv(data_folder_1 + 
                            'binary_reduced/student_ids.csv', header=None)
    
    stud_ids_1 = ids.sample(frac=0.5)
    stud_ids_2 = ids.drop(stud_ids_1.index)
    
    # create data frame of stud_ids_1 that are also in stud_ids
    group_1_ids = pd.merge(stud_ids, stud_ids_1, how='inner', on=0)
    group_2_ids = pd.merge(stud_ids, stud_ids_2, how='inner', on=0)
    return group_1_ids, group_2_ids    


def dif_analysis(degree_1, degree_2, course_id, group_1_ids, group_2_ids, verbose=True):
    '''
    
    '''

    MISSING_VALUE = -99999
    parent_dir = os.path.dirname(os.getcwd())
    analysis_dir = parent_dir + '/src/analysis/'
    sys.path.append(analysis_dir)
    from dif import compare_abilities
    from dif import log_dif_analysis
    from dif import stud_ids_by_course
    # choose degree and declare groups to be compared


    if degree_2 is None:
        degree = degree_1
    else:
        degree = degree_1 + '_' + degree_2

    data_folder = parent_dir + '/data/real/' + degree + '/'
    stud_ids = pd.read_csv(data_folder + 
                            'binary_reduced/student_ids.csv', header=None)
    course_ids = pd.read_csv(data_folder + 
                            'binary_reduced/item_ids.csv', header=None)
    #print(course_ids.shape)
    # Here you can write your own code to extract student groups to be compared. One Example already implemented: student ids by course name.
    #random_course = 'course_10'
    
    if group_1_ids is None or group_2_ids is None and course_id is not None:
        group_1_ids, group_2_ids = stud_ids_by_course(degree_1 = degree_1, degree_2 = degree_2, course_id = course_id)
        # columns need to be named 'student_id'
        group_1_ids.columns = ['student_id']
        group_2_ids.columns = ['student_id']
        df_res = compare_abilities(degree, group_1_ids, group_2_ids, plot=verbose)



    # Perform the Logisitc Regression DIF test
    if group_2_ids is None and group_1_ids is None:
        #print a warning
        raise ValueError('group_ids need to be declared in order to run a dif analysis')
    else:
        try:
            dif_results = log_dif_analysis(degree_1, degree_2, group_1_ids, group_2_ids, plot=verbose)
        except Exception as e:
            print('An error occured in dif analysis: ', e)
            print(traceback.format_exc())



    try:
        # This cell searches for the course with highest difference between the two groups for a single course:
        max_mean = 0
        max_course = None
        if course_id is None:
            for course in course_ids.values:
                #print(course)
                try:
                    group_1_ids, group_2_ids = stud_ids_by_course(degree_1 = degree_1, degree_2 = degree_2, course_id = course[0])
                    
                except ValueError:
                    continue

                # columns need to be named 'student_id'
                group_1_ids.columns = ['student_id']
                group_2_ids.columns = ['student_id']

                df_res = compare_abilities(degree, group_1_ids, group_2_ids, plot=False)
                if max_mean < df_res['mean_dif'].abs().mean():
                    max_mean = df_res['mean_dif'].abs().mean()
                    max_course = course[0]
                    max_df = df_res
            if max_course is not None:
                group_1_ids, group_2_ids = stud_ids_by_course(degree_1 = degree_1, degree_2 = degree_2, course_id = max_course)
                group_1_ids.columns = ['student_id']
                group_2_ids.columns = ['student_id']

                print('--------------'  +max_course+  '--------------')
                df_res = compare_abilities(degree, group_1_ids, group_2_ids, plot=verbose)
    except Exception as e:
        print('An error occured in ability comparrison: ', e)


