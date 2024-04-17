import warnings
import sys
import importlib
import matplotlib.pyplot as plt
import numpy as np
warnings.filterwarnings("ignore")
import pandas as pd
import importlib
import pickle

def concat_taggedRInput(input_1, input_2, err_crit = True):
    # Check input_1 and input_2 for same item_ids and times
    # Get row names
    row_names_1 = input_1.index.values
    row_names_2 = input_2.index.values

    # Get intersectoin of column names
    inter_items = list(set(row_names_1) & set(row_names_2))

    if len(inter_items) == 0:
        if err_crit:
            raise ValueError("No common items found in the two degrees. Do Analysis seperately to save time.")
    # else:
    #     print('No. of item_ids intersecting: ', len(np.unique(inter_items)))
    #     # Append the columns 

    

    # set the index to be the index + ' ' + time
    input_1.index = input_1.index + ' _iden_ ' + input_1['time']
    input_2.index = input_2.index + ' _iden_ ' + input_2['time']

    # Drop the time column
    input_1 = input_1.drop(['time'], axis = 1)
    input_2 = input_2.drop(['time'], axis = 1)
    
    # Concatenate the two dataframes
    input_1 = pd.concat([input_1, input_2], axis = 1)

    # If a Column appears twice, then concatenate the two columns into one by replacing the nan values with the non-nan values if any
    # Get the column names
    col_names = input_1.columns.values

    # Get the duplicate column names
    duplicate_col_names = [x for x in np.unique(col_names) if len(np.where(col_names==x)[0]) > 1]
    #print(col_names)
    # Iterate over the duplicate column names
    for col_name in duplicate_col_names:
        duplicate = input_1[col_name]
        # split the duplicate into two seperate columns
        duplicate_1 = duplicate.iloc[:,0]
        duplicate_2 = duplicate.iloc[:,1]
        # Delete all column in input_1 with the name col_name
        input_1 = input_1.drop([col_name], axis = 1)
        # Concatenate the two columns
        duplicate_1 = duplicate_1.fillna(duplicate_2)
        # Add the column to input_1
        input_1[col_name] = duplicate_1

    # Extradct the time column from the index and set it as a column in the dataframe at the beginning
    time = input_1.index.str.split(' _iden_ ').str[1]
    input_1.index = input_1.index.str.split(' _iden_ ').str[0]
    input_1.insert(0, 'time', time)

    # Change all nan values to 0
    input_1 = input_1.fillna(-99999)
    return input_1


def irt_analysis(degree_1, pass_grades, fail_grades, degree_2 = None, nm_rates=0.05, data_format = False, cut_off_search = False):
    
    import os
    cwd = os.getcwd()
    # Get the directory name from the current working directory
    dirname = os.path.basename(cwd)
    sys.path.append('../src')
    prepr_folder = os.path.join(os.path.dirname(cwd),'src', 'preprocessing')
    sys.path.append(prepr_folder)
    plotting_folder = os.path.join(os.path.dirname(cwd),'src', 'plotting')
    sys.path.append(plotting_folder)

    import Classes
    from preprocessing import StudentGroup_to_StudentGroup
    from preprocessing import StudentGroup_to_taggedRInput
    from preprocessing import taggedRInput_to_taggedRInput
    importlib.reload(StudentGroup_to_StudentGroup)
    importlib.reload(StudentGroup_to_taggedRInput)
    importlib.reload(Classes)
    
    # sort degree string names
    if degree_2 is not None:
        sorted_degs = np.sort([degree_1, degree_2])
        
        degree_1 = sorted_degs[0]
        degree_2 = sorted_degs[1]

    # get list of all folders in the data folder
    deg_folders = os.path.join(os.path.dirname(cwd),'data/real')
    deg_folders = os.listdir(deg_folders)



    # Data variants
    variants = ['binary', 'non_binary', 'binary_reduced', 'non_binary_reduced']
                
    if degree_2 is not None:
        
        if not os.path.exists(os.path.join(os.path.dirname(cwd),'data/real', degree_1)):
            # Throw error that degree files or folder does not exist
            raise ValueError('The degree folder does not exist.')
        if not os.path.exists(os.path.join(os.path.dirname(cwd),'data/real', degree_2)):
            # Throw error that degree files or folder does not exist
            raise ValueError('The degree_2 folder does not exist.')
        degree = degree_1 + '_' + degree_2
        multi_degree = True
    else:
        multi_degree = False
        degree = degree_1
    # Build degree folder if it does not exist
    if not os.path.exists(os.path.join(os.path.dirname(cwd),'data/real', degree)):
        os.makedirs(os.path.join(os.path.dirname(cwd),'data/real', degree))

    # Construct the path to the data files
    data_folder = os.path.join(os.path.dirname(cwd),'data/real', degree)

    # Build folders if they do not exist
    for variant in variants:
        if not os.path.exists(os.path.join(data_folder, variant)):
            os.makedirs(os.path.join(data_folder, variant))
    if not os.path.exists(os.path.join(data_folder, 'plots')):
        os.makedirs(os.path.join(data_folder, 'plots'))
    if not os.path.exists(os.path.join(data_folder,  'irt_results')):
        os.makedirs(os.path.join(data_folder, 'irt_results'))
    if not os.path.exists(os.path.join(data_folder, 'irt_results', '1pl_1dim')):
        os.makedirs(os.path.join(data_folder, 'irt_results', '1pl_1dim'))
    if not os.path.exists(os.path.join(data_folder, 'irt_results', '2pl_1dim')):
        os.makedirs(os.path.join(data_folder, 'irt_results', '2pl_1dim'))
    if not os.path.exists(os.path.join(data_folder, 'irt_results', '2pl_2dim')):
        os.makedirs(os.path.join(data_folder, 'irt_results', '2pl_2dim'))
    if not os.path.exists(os.path.join(data_folder, 'irt_results', '2pl_3dim')):
        os.makedirs(os.path.join(data_folder, 'irt_results', '2pl_3dim'))
    if not os.path.exists(os.path.join(data_folder, 'irt_results', 'model_selection')):
        os.makedirs(os.path.join(data_folder, 'irt_results', 'model_selection'))
    if not os.path.exists(os.path.join(data_folder, 'irt_results', 'models')):
        os.makedirs(os.path.join(data_folder, 'irt_results', 'models'))
    if not os.path.exists(os.path.join(data_folder, 'irt_results', 'models', 'model_loadings')):
        os.makedirs(os.path.join(data_folder, 'irt_results', 'models', 'model_loadings'))
    # Delete course_names.csv if it exists
    if os.path.exists(os.path.join(data_folder, 'course_names.csv')):
        os.remove(os.path.join(data_folder, 'course_names.csv'))

    # Check the data 
    if not data_format and degree_2==False:
        import re

        tagged_input = pd.read_csv(os.path.join(data_folder, 'taggedRInput' + '.csv'), index_col='COURSE', header=0)

        # copy of tagged_input wihtout first column
        tagged_input_copy = tagged_input.iloc[:, 1:].copy()

        # assert if the first column is not called 'time'
        assert tagged_input.columns[0] == 'time', 'The second column should be called time'

        # assert if the elements in the time column are not strings
        assert all(isinstance(x, str) for x in tagged_input['time']), 'The time column is not a string, all course names should be strings'

        # assert if 'NaN' values are present in the time column
        assert not np.any(tagged_input['time'] == 'NaN'), 'There are NaN string values in the time column.'

        # assert if the elements of the time column do not begin with 'W' or 'S'
        assert all(re.match(r'[WS]', x) for x in tagged_input['time']), 'The time column does not contain strings that begin with W or S.'

        # assert if the elements of the time column do exceed 8 characters
        assert all(len(x) <= 8 for x in tagged_input['time']), 'The time column contains strings that exceed 8 characters. It is highly likeli that some course names include seperators in the name.'

        index = [x+tagged_input['time'][i] for i,x in enumerate(tagged_input_copy.index)]

        # assert if there are douplicates in the index 
        assert len(index) == len(np.unique(index)), 'There are duplicates in the index+time combination.'

        # assert if index are not strings   
        assert all(isinstance(x, str) for x in index), 'The index is not a string, all course names should be strings'

        # assert if NaN or 'NaN' values are present in the index
        assert not np.any(index == 'NaN'), 'There are NaN string values in the index, which relates to the course name'

        # assert if there are douplicates in the columns
        columns = tagged_input_copy.columns
        assert len(columns) == len(np.unique(columns)), 'There are duplicates in the columns'

        # assert if NaN or 'NaN' values are present in the columns
        assert not np.any(columns == 'NaN'), 'There are NaN string values in the columns, which relates to the student name'


    if not data_format:
        if not multi_degree:
            tagged_input = pd.read_csv(os.path.join(data_folder, 'taggedRInput' + '.csv'), index_col=0, header=0)
        else: 
            data_folder_1 = os.path.join(os.path.dirname(cwd),'data/real', degree_1)
            data_folder_2 = os.path.join(os.path.dirname(cwd),'data/real', degree_2)
            tagged_input_1 = pd.read_csv(os.path.join(data_folder_1, 'taggedRInput' + '.csv'), index_col=0, header=0)
            tagged_input_2 = pd.read_csv(os.path.join(data_folder_2, 'taggedRInput' + '.csv'), index_col=0, header=0)

            stud_ids_1 = tagged_input_1.columns.values[1:]
            stud_ids_2 = tagged_input_2.columns.values[1:]
            
    
            tagged_input = concat_taggedRInput(tagged_input_1, tagged_input_2, err_crit=False)



            
            # save the concatenated tagged_input as taggedRInput.csv
            tagged_input.to_csv(os.path.join(data_folder, 'taggedRInput' + '.csv'), index=True, header=True)
        # copy of tagged_input wihtout first column
        tagged_input_copy = tagged_input.iloc[:, 1:].copy()
        index = tagged_input_copy.index
        # go through each row of the dataframe
        

        # Define a function to process each course group
        def process_course_group(course_group):
            # Replace -99999 with NaN and round up
            course_group = course_group.replace(-99999, np.nan).apply(np.ceil)

            # Compute the mean, skipping NaN values
            course_mean = course_group.mean(axis=0, skipna=True)

    
            # Replace NaN values with -99999
            return course_mean.fillna(-99999)

        if not multi_degree:
            ellbow_data = pd.DataFrame(index=np.unique(index), columns=tagged_input_copy.columns)

            # Group by course and apply the processing function
            ellbow_data = tagged_input_copy.groupby(level=0).apply(process_course_group)
            
            # for course in np.unique(index):
            #     # find all rows with course as index
            #     course_rows = tagged_input_copy.loc[course]

            #     # Replace -99999 values with NaN
            #     course_rows.replace(-99999, np.nan, inplace=True)

            #     # Round up to next integer 
            #     course_rows = course_rows.apply(np.ceil, skipna=True)


            #     # Compute the mean skipping NaN values
            #     if len(np.shape(course_rows)) > 1:
            #         course_mean = course_rows.mean(axis=0, skipna=True)
            #     else:
            #         course_mean = course_rows
                    

            #     # Replace NaN values with -99999
            #     course_mean.replace(np.nan, -99999, inplace=True)
                

            #     # Replace the original rows with the mean
            #     ellbow_data.loc[course] = course_mean

            # for each row, count number of non -99999
            num_non_miss = ellbow_data.apply(lambda x: x[x != -99999].count(), axis=1)
            num_miss = ellbow_data.apply(lambda x: x[x == -99999].count(), axis=1)
            non_missing_rates = (num_non_miss/(num_miss+num_non_miss)).sort_values(ascending=False)
            non_missing_rates = non_missing_rates[non_missing_rates>nm_rates]
            plt.plot(np.arange(len(non_missing_rates)),non_missing_rates, label='Fraction of non-missing values per course')
            plt.savefig(data_folder + '/plots/missing_values_analysis.pdf', format="pdf")
            course_names = pd.DataFrame()
            names = np.array(non_missing_rates.index)
            course_names['Modul'] = names
            course_names.to_csv(os.path.join(data_folder, 'course_names.csv'), index=False)
        else:
            # check if course_names.csv exists
            if os.path.exists(os.path.join(data_folder_1, 'course_names.csv')) and os.path.exists(os.path.join(data_folder_2, 'course_names.csv')):
                names_1 = pd.read_csv(os.path.join(data_folder_1, 'course_names.csv'))['Modul'].values
                names_2 = pd.read_csv(os.path.join(data_folder_2, 'course_names.csv'))['Modul'].values
                course_names_1 = pd.DataFrame()
                course_names_1['Modul'] = names_1
                course_names_2 = pd.DataFrame()
                course_names_2['Modul'] = names_2
                print('Number of courses in degree 1: ', course_names_1.shape[0])
                print('Number of courses in degree 2: ', course_names_2.shape[0])
                course_names = pd.concat([course_names_1, course_names_2], axis=0)
                course_names['Modul'] = course_names['Modul'].str.split(' _iden_ ').str[0]
              
                course_names = course_names.drop_duplicates()
                course_names.to_csv(os.path.join(data_folder, 'course_names.csv'), index=False)
                # print the number of courses that are in both degrees
                print('Number of courses in intersection: ', course_names_1.shape[0] + course_names_2.shape[0] -course_names.shape[0])
                
            else:
                # Warn that course_names for single majors are not available
                warnings.warn('Course names for single majors are not available. It is recommended to run the analysis for single majors first.')
                tagged_input_1_copy = tagged_input_1.iloc[:, 1:].copy()
                tagged_input_2_copy = tagged_input_2.iloc[:, 1:].copy()

                # print the number of courses in each degree
                print('Number of courses in degree 1: ', tagged_input_1_copy.shape[0])
                print('Number of courses in degree 2: ', tagged_input_2_copy.shape[0])

                ellbow_data_1 = pd.DataFrame(index=np.unique(index), columns=tagged_input_1_copy.columns)
                ellbow_data_2 = pd.DataFrame(index=np.unique(index), columns=tagged_input_2_copy.columns)

                

                ellbow_data_1 = tagged_input_1_copy.groupby(level=0).apply(process_course_group)
                # for course in np.unique(tagged_input_1_copy.index):
                #     # find all rows with course as index
                #     course_rows = tagged_input_1_copy.loc[course]

                #     # Replace -99999 values with NaN
                #     course_rows.replace(-99999, np.nan, inplace=True)

                #     # Round up to next integer 
                #     course_rows = course_rows.apply(np.ceil, skipna=True)

                #     # Compute the mean skipping NaN values
                #     if len(np.shape(course_rows)) > 1:
                #         course_mean = course_rows.mean(axis=0, skipna=True)
                #     else:
                #         course_mean = course_rows
                        

                #     # Replace NaN values with -99999
                #     course_mean.replace(np.nan, -99999, inplace=True)
                    

                #     # Replace the original rows with the mean
                #     ellbow_data_1.loc[course] = course_mean
                
                # for each row, count number of non -99999
                num_non_miss = ellbow_data_1.apply(lambda x: x[x != -99999].count(), axis=1)
                num_miss = ellbow_data_1.apply(lambda x: x[x == -99999].count(), axis=1)
                non_missing_rates = (num_non_miss/(num_miss+num_non_miss)).sort_values(ascending=False)
                non_missing_rates = non_missing_rates[non_missing_rates>nm_rates]
                course_names_1 = pd.DataFrame()
                names = np.array(non_missing_rates.index)
                course_names_1['Modul'] = names_1

                # for course in np.unique(tagged_input_2_copy.index):
                #     # find all rows with course as index
                #     course_rows = tagged_input_2_copy.loc[course]

                #     # Replace -99999 values with NaN
                #     course_rows.replace(-99999, np.nan, inplace=True)

                #     # Round up to next integer 
                #     course_rows = course_rows.apply(np.ceil, skipna=True)

                #     # Compute the mean skipping NaN values
                #     if len(np.shape(course_rows)) > 1:
                #         course_mean = course_rows.mean(axis=0, skipna=True)
                #     else:
                #         course_mean = course_rows
                        

                #     # Replace NaN values with -99999
                #     course_mean.replace(np.nan, -99999, inplace=True)
                    

                #     # Replace the original rows with the mean
                #     ellbow_data_2.loc[course] = course_mean
                ellbow_data_2 = tagged_input_2_copy.groupby(level=0).apply(process_course_group)
                # for each row, count number of non -99999
                num_non_miss = ellbow_data_2.apply(lambda x: x[x != -99999].count(), axis=1)
                num_miss = ellbow_data_2.apply(lambda x: x[x == -99999].count(), axis=1)
                non_missing_rates = (num_non_miss/(num_miss+num_non_miss)).sort_values(ascending=False)
                non_missing_rates = non_missing_rates[non_missing_rates>nm_rates]
                course_names_2 = pd.DataFrame()
                names = np.array(non_missing_rates.index)
                course_names_2['Modul'] = names_2
                #merge course names
                # print the number of courses in each degree
                print('Number of courses in degree 1 after missing value filter: ', course_names_1.shape[0])
                print('Number of courses in degree 2 after missing value filter: ', course_names_2.shape[0])

                course_names = pd.concat([course_names_1, course_names_2], axis=0)
                course_names['Modul'] = course_names['Modul'].str.split(' _iden_ ').str[0]
                course_names = course_names.drop_duplicates()
                course_names.to_csv(os.path.join(data_folder, 'course_names.csv'), index=False)



        # # Calculate unique courses once
        # unique_courses = np.unique(index)

        # # Function to process each group (course)
        # def process_course_group(course_group):
        #     # Replace -99999 with NaN, round up, and compute mean
        #     course_mean = course_group.replace(-99999, np.nan).apply(np.ceil).mean(skipna=True)

        #     # Replace NaN with -99999
        #     return course_mean.replace(np.nan, -99999)

        # # Process data based on 'multi_degree' value
        # if not multi_degree:
        #     # Group by course and process each group
        #     ellbow_data = tagged_input_copy.groupby(level=0).apply(process_course_group)
        # else:
        #     # Process each dataset separately
        #     ellbow_data_1 = tagged_input_1.iloc[:, 1:].groupby(level=0).apply(process_course_group)
        #     ellbow_data_2 = tagged_input_2.iloc[:, 1:].groupby(level=0).apply(process_course_group)

        #     # Merge course names and handle duplicates
        #     course_names = pd.concat([ellbow_data_1, ellbow_data_2], axis=0)
        #     course_names['Modul'] = course_names['Modul'].str.split(' _iden_ ').str[0]
        #     course_names = course_names.drop_duplicates()

        # # Count non-missing values and calculate rates
        # non_missing_counts = ellbow_data.apply(lambda x: (x != -99999).sum(), axis=1)
        # missing_counts = ellbow_data.apply(lambda x: (x == -99999).sum(), axis=1)
        # non_missing_rates = (non_missing_counts / (missing_counts + non_missing_counts)).sort_values(ascending=False)
        # non_missing_rates = non_missing_rates[non_missing_rates > nm_rates]

        # # Plotting
        # plt.plot(np.arange(len(non_missing_rates)), non_missing_rates, label='Fraction of non-missing values per course')
        # plt.savefig(os.path.join(data_folder, 'plots/missing_values_analysis.pdf'), format="pdf")

        # # Save course names
        # course_names = pd.DataFrame(non_missing_rates.index, columns=['Modul'])
        # course_names.to_csv(os.path.join(data_folder, 'course_names.csv'), index=False)

    # If data is in raw format, preprocess it and save it as a StudentGroup object in a pickle file
    importlib.reload(StudentGroup_to_StudentGroup)
    importlib.reload(taggedRInput_to_taggedRInput)
    if data_format:
        print('Data is in raw format ...')
        if multi_degree:
            # Throw error if multi_degree is True and data_format is True
            raise ValueError('multi_degree and data_format cannot be True at the same time, since procedure is only implemented for tagged data!')
        # Read Raw data
        # ....

        # Preprocess raw data
        # ....

        # Save preprocessed raw data as StudentGroup object in pickle file
        # ....
        placeholder=0

        # Read StudentGroup object from pickle file
        student_group = pickle.load(open(os.path.join(data_folder, 'StudentGroup.pickle'), 'rb'))


        #for student in student_group.students:
        #    for c, course in enumerate(student.courseNames):
        #        if course == 'Industrial Management':
        #            # Drop the index from all student arrays:
        #            student.pop(c)

        # Preprocess StudentGroup object
        
        student_group,_,_ = StudentGroup_to_StudentGroup.group_filter(student_group, 
                                                                pass_grades=pass_grades,
                                                                fail_grades=fail_grades, 
                                                                verbose=True,
                                                                cut_off_search=cut_off_search,
                                                                course_names = course_names)
        # Save preprocessed StudentGroup object in pickle file
        pickle.dump(student_group, open(os.path.join(data_folder, 'StudentGroup_filtered.pickle'), 'wb'))
        print('      \n','... preprocessed StudentGroup object saved in pickle file')
        
        # Check if course_names.csv exists
        if not os.path.exists(os.path.join(data_folder, 'course_names.csv')):
            # If not, create it
            course_names = pd.DataFrame()
            names = []
            for student in student_group.students:
                for course in student.courseNames:
                    if course not in names:
                        names.append(course)
            course_names['Modul'] = names
            course_names.to_csv(os.path.join(data_folder, 'course_names.csv'), index=False)
        
    else:
        print('Data is already in tagged format ...')
        print('... taggedRInput.csv is loaded from csv file')
        if multi_degree:
            tagged_input = pd.read_csv(os.path.join(data_folder, 'taggedRInput' + '.csv'), index_col=0, header=0)
            # # If multi_degree is True we append the data from degree_2 under the data from degree_1
            # data_folder_1 = os.path.join(os.path.dirname(cwd),'data/real', degree_1)
            # data_folder_2 = os.path.join(os.path.dirname(cwd),'data/real', degree_2)
            # tagged_input_1 = pd.read_csv(os.path.join(data_folder_1, 'taggedRInput' + '.csv'), index_col=0, header=0)
            # tagged_input_2 = pd.read_csv(os.path.join(data_folder_2, 'taggedRInput' + '.csv'), index_col=0, header=0)
            # #print(tagged_input_1.head())
            # tagged_input = concat_taggedRInput(tagged_input_1, tagged_input_2, err_crit=False)
        else:
            tagged_input = pd.read_csv(os.path.join(data_folder, 'taggedRInput' + '.csv'), index_col=0, header=0)

        # Save a copy of the tagges input as a csv file in the data folder
        tagged_input.to_csv(os.path.join(data_folder, 'taggedRInput_unprocessed' + '.csv'))

        #print(len(course_names['Modul'].values))

        print('... taggedRInput.csv is converted to StudentGroup object and preprocessing is applied ...')
        #print(course_names['Modul'].values)
        pass_grades, fail_grades = taggedRInput_to_taggedRInput.taggedRInput_to_preprocessing(tagged_input, 
                                                                                            pass_grades, 
                                                                                            fail_grades, 
                                                                                            degree, 
                                                                                            verbose=True,
                                                                                            cut_off_search=cut_off_search,
                                                                                            course_names = course_names['Modul'].values)
    # check wether student_ids exist in both degrees
    if multi_degree: 
        id_folder_1 = data_folder_1 + '/binary_reduced'
        if not os.path.exists(id_folder_1):
            os.makedirs(id_folder_1)
        if not os.path.exists(id_folder_1 + '/student_ids.csv'):
            #read StudentGroup_filtered.pickle
            student_group = pickle.load(open(os.path.join(data_folder, 'StudentGroup_filtered.pickle'), 'rb'))
            # get student_ids
            student_ids = [student.name for student in student_group.students]
            #intersect with student_ids from degree_1 
            student_ids_1 = np.intersect1d(student_ids, stud_ids_1)
            #save student_ids_1 using pandas
            pd.DataFrame(student_ids_1).to_csv(id_folder_1 + '/student_ids.csv', header=False, index=False)

        id_folder_2 = data_folder_2 + '/binary_reduced'
        if not os.path.exists(id_folder_2):
            os.makedirs(id_folder_2)
        if not os.path.exists(id_folder_2 + '/student_ids.csv'):
            #read StudentGroup_filtered.pickle
            student_group = pickle.load(open(os.path.join(data_folder, 'StudentGroup_filtered.pickle'), 'rb'))
            # get student_ids
            student_ids = [student.name for student in student_group.students]
            #intersect with student_ids from degree_1 
            student_ids_2 = np.intersect1d(student_ids, stud_ids_2)
            #save student_ids_1 using pandas
            pd.DataFrame(student_ids_2).to_csv(id_folder_2 + '/student_ids.csv', header=False, index=False)  

    # Nothing to adjust in this cell! Just run it!

    # importlib.reload(StudentGroup_to_taggedRInput)
    # importlib.reload(taggedRInput_to_taggedRInput)

    if data_format or not data_format:
        # Read preprocessed StudentGroup object from pickle file
        student_group = pickle.load(open(os.path.join(data_folder, 'StudentGroup_filtered.pickle'), 'rb'))
        print(len(student_group.students))
        #if not os.path.exists(os.path.join(data_folder, 'course_names.csv')):
            # Use a set for efficient uniqueness checking
        course_names_set = set()

        # Iterate through students and add their courses to the set
        for student in student_group.students:
            course_names_set.update(student.courseNames)

        # Convert the set to a DataFrame
        course_names = pd.DataFrame({'Modul': list(course_names_set)})

        # Save to CSV
        course_names.to_csv(os.path.join(data_folder, 'course_names.csv'), index=False)
        
        
        #     # If not, create it
        #     course_names = pd.DataFrame()
        #     names = []
        #     for student in student_group.students:
        #         for course in student.courseNames:
        #             if course not in names:
        #                 names.append(course)
        #     course_names['Modul'] = names
        #     course_names.to_csv(os.path.join(data_folder, 'course_names.csv'), index=False)
    
        # Preprocess StudentGroup object into IRT format
        #vars_df, vars_co  = StudentGroup_to_taggedRInput.StudentGroup_to_df(student_group, fail_grades, pass_grades)



        # Save preprocessed StudentGroup object in csv file where the columns are the items and the rows are the students
        # for v, var in enumerate(variants):
        #     print(vars_df[v].shape)
        #     # Set the row names of the pandas dataframe
        #     vars_df[v].index = vars_co[v]

        #     # Write the pandas dataframe to a csv file where the columns are the column_names and the rows are the row_names
        #     vars_df[v].to_csv(os.path.join(data_folder, str(var) + '/' 'taggedRInput' + '.csv'), index=True, header=True)
        
        #     # Write the numpy array course offerings to a csv file
        #     np.savetxt(os.path.join(data_folder, str(var) + '/' 'item_ids' + '.csv'), vars_co[v], delimiter=",", fmt='%s')

        #for df in vars_df:
        #    print(df.shape)
        ##    for col in df.transpose(): 

    # Nothing to adjust in this cell! Just run it!

    importlib.reload(StudentGroup_to_taggedRInput)
    import taggedRInput_to_taggedRInput
    importlib.reload(taggedRInput_to_taggedRInput)

    print('StudentGroup object is transformed into taggedRInput format ...') 
    # Read preprocessed StudentGroup object from pickle file
    # student_group = pickle.load(open(os.path.join(data_folder, 'StudentGroup_filtered.pickle'), 'rb'))
    # print(len(student_group.students))

    # Preprocess StudentGroup object into IRT format
    vars_df, vars_co  = StudentGroup_to_taggedRInput.StudentGroup_to_df(student_group, fail_grades, pass_grades)

    # Save preprocessed taggedRInput in csv file
    for v, var in enumerate(variants):
        vars_df[v].to_csv(os.path.join(data_folder, str(var) + '/' 'taggedRInput' + '.csv'), header=False,
            index=False)
        pd.DataFrame([student.name for student in student_group.students]).to_csv(os.path.join(data_folder, str(var) + '/' 'student_ids' + '.csv'), header=False,
                    index=False)
        print(len(pd.DataFrame(vars_df[v].columns.values)))
        # Write the numpy array course offerings to a csv file
        np.savetxt(os.path.join(data_folder, str(var) + '/' 'item_ids' + '.csv'), vars_co[v], delimiter=",", fmt='%s')
        print('      \n','... preprocessed taggedRInput of variant: ' + str(var)  + ' saved in csv file')


    # Nothing to adjust in this cell! Just run it!

    for var in variants:
        # Read data from csv file
        df = pd.read_csv(os.path.join(data_folder, str(var) + '/' 'taggedRInput' + '.csv'), header=None)

        # Calculate pass rates for each CO, only include 0,1 entries
        pass_rates = df.replace(-99999, np.nan).mean(axis=1).dropna()

        # Save pass rates in csv file
        pass_rates.to_csv(data_folder + '/' +  var + '/pass_rates.csv', header=True, index=True)

        # Calculate gpas for each student, only include non -99999 entries
        gpas = df.replace(-99999, np.nan).mean(axis=0).dropna()

        # Save gpas in csv file
        gpas.to_csv(data_folder + '/' +  var + '/gpas.csv', header=True, index=True)

        
        if var == 'binary':
            bin_df = df

    # check wether q3 results exist already
    def delete_outlier_files(model):
        if os.path.exists(os.path.join(data_folder, 'irt_results', model,'q3_outliers.csv')):
            # delete q3 results
            os.remove(os.path.join(data_folder, 'irt_results', model,'q3_outliers.csv'))
            os.remove(os.path.join(data_folder, 'irt_results', model,'q3_outliers_values.csv'))
        return

    delete_outlier_files('1pl_1dim')
    delete_outlier_files('2pl_1dim')
    delete_outlier_files('2pl_2dim')
    delete_outlier_files('2pl_3dim')

    # Nothing to adjust in this cell! Just run it!

    import os
    # Get parent directory
    analysis_dir = os.path.dirname(os.getcwd()) + '/src/analysis/'


    # Enables the %%R magic, not necessary if you've already done this
    #%load_ext rpy2.ipython
    #%reload_ext rpy2.ipython
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr

    # load R script
    r = robjects.r
    r.source(analysis_dir + 'models.R')
    r['compute_models'](degree)

    # Nothing to adjust in this cell! Just run it!

    import os
    # Get parent directory
    analysis_dir = os.path.dirname(os.getcwd()) + '/src/analysis/'


    # Enables the %%R magic, not necessary if you've already done this
    #%load_ext rpy2.ipython
    #%reload_ext rpy2.ipython
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr
    # load R script
    r = robjects.r

    # load R script for imputation choose from 'mean' or 'mipca'
    r = robjects.r
    r.source(analysis_dir + 'imputation.R')
    r['imputation'](degree, imputation_method='mean')


    # Nothing to adjust in this cell! Just run it!

    import os
    # Get parent directory
    analysis_dir = os.path.dirname(os.getcwd()) + '/src/analysis/'


    # Enables the %%R magic, not necessary if you've already done this
    #%load_ext rpy2.ipython
    #%reload_ext rpy2.ipython
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr
    # load R script
    r = robjects.r

    # load R script for imputation choose from 'mean' or 'mipca'
    r = robjects.r
    r.source(analysis_dir + 'get_variance.R')
    r['compute_model_loadings'](degree)

    # Nothing to adjust in this cell! Just run it!

    # Model Selection Results
    anova_df = pd.read_csv(data_folder + '/irt_results/model_selection/anova_1PL_2PL_1DIM.csv', index_col=0)

    # Plot Anova results as table
    print(anova_df)
    # Save anova results as table
    anova_df.to_csv(data_folder + '/plots/anova.csv')

    # Plot screeplot
    scree = pd.read_csv(data_folder + '/irt_results/model_selection/scree_data.csv', index_col=0)
    plt.figure(figsize=(8.5,5))
    plt.plot(np.arange(len(scree))[:10], scree[:10], alpha=0.5, color='black', linestyle='dotted')
    plt.scatter(np.arange(len(scree))[:10], scree[:10], color='black', alpha=0.7)
    plt.xticks(np.arange(len(scree))[:10],np.arange(len(scree))[:10])
    plt.xlabel('number of components')
    plt.ylabel('eigenvalues')
    plt.title("Scree Plot")
    plt.savefig(data_folder + '/plots/scree_plot.pdf', format="pdf")
    plt.show() 

    # Q3 value printing:
    def print_q3(model):
        # read binary reduced item ids
        bin_c = np.array(pd.read_csv(data_folder + '/binary_reduced/item_ids.csv', header=None).values).flatten()
        #print(bin_c)
        q3_values = pd.read_csv(data_folder + '/irt_results/' + model + '/q3_outliers_values.csv', sep=',')
        q3_inds = pd.read_csv(data_folder + '/irt_results/' + model + '/q3_outliers.csv')

        
        c_pairs = [(bin_c[int(c1)-1],bin_c[int(c2)-1]) for c1,c2 in zip(q3_inds['row'], q3_inds['col'])]

        
        c_pair_vals = q3_values['combined_outliers_values']

        print(model + ' Q3 values:')
        for c_pair, c_pair_val in zip(c_pairs, c_pair_vals):
            print(str(c_pair[0]) + ' - ' + str(c_pair[1]) + ': ' + str(c_pair_val))
        print('\n')
        return

    #print_q3('1pl_1dim')
    #print_q3('2pl_1dim')
    #print_q3('2pl_2dim')
    #print_q3('2pl_3dim')


    # choose best fitting model:
    from collections import Counter

    # Criteria to evaluate
    criteria = ['AIC', 'SABIC', 'HQ', 'BIC']

    # set the index of the anova_df to the column 'model'
    df = pd.DataFrame(anova_df)
    df['model'] = df.index

    # List to store the best models based on each criterion
    best_models_list = []

    print(df['BIC'])

    # Iterate through each criterion and get the best model
    for criterion in criteria:
        best_model, _ = df[df[criterion] == df[criterion].min()]['model'].iloc[0], df[criterion].min()
        best_models_list.append(best_model)

    # Count how many times each model is selected as best
    counter = Counter(best_models_list)

    # Get the model that is most frequently selected as best
    best_overall_model, freq = counter.most_common(1)[0]

    # Print out the best models for each criterion
    print(f"Best models based on each criterion are: {best_models_list}")




    best_overall_model, _ = df[df['BIC'] == df['BIC'].min()]['model'].iloc[0], df['BIC'].min()
    print("Use BIC for model selection for know: " + str(best_overall_model))

    # Print out the overall best model
    print(f"The overall best model is according to BIC: {best_overall_model}")

    import loadings
    importlib.reload(loadings)
    loadings.plot_loadings(degree, best_overall_model)


    # Nothing to adjust in this cell! Just run it!

    # Validity of Estimates
    save_dir = os.path.dirname(os.getcwd()) + '/data/real/'+ degree+'/plots/'
    verbose = True
    # validity  
    import validity
    importlib.reload(validity)
    validity.plot_validity(degree, best_overall_model, verbose=verbose, save_path=save_dir)


    # Regression Validation:

    save_dir = os.path.dirname(os.getcwd()) + '/data/real/'+ degree+'/plots/'
    verbose = True
    # validity  
    import validity
    importlib.reload(validity)
    validity.regression_validation(degree, best_overall_model, verbose=verbose, save_path=save_dir)


    # Nothing to adjust in this cell! Just run it!

    # Reliability of Estimates
    # reliability
    import reliability
    importlib.reload(reliability)
    reliability.plot_reliability(degree, best_overall_model)    


    import difficulty
    importlib.reload(difficulty)
    degree_names = [degree_1, degree_2]
    diff_df = difficulty.plot_difficulty(best_model, data_folder, degree_names, color_gradient=False) 



    import ability 
    importlib.reload(ability)
    if degree_2 != None:
        ability.plot_ability(best_model, data_folder, degree_names) 

    return