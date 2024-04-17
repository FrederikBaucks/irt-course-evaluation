import numpy as np
import pandas as pd
import sys
import os
import StudentGroup_to_StudentGroup
import StudentGroup_to_taggedRInput
import pickle
import importlib
importlib.reload(StudentGroup_to_StudentGroup)
# Get the parent directory of the current file
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Add the parent directory to sys.path
sys.path.append(parent_dir)
# Import the module from the parental folder
import Classes
import re

def taggedRInput_to_variants(tagged_input, fail_grades, pass_grades):
    '''
    Convert a taggedRInput object to a list of data frames, where each data frame is a variant of the taggedRInput object.  
    Parameters
    ----------
    tagged_input : taggedRInput
        The taggedRInput object to be converted to a data frame.
    fail_grades : list
        A list of grades that indicate that a student failed a course.
    pass_grades : list
        A list of grades that indicate that a student passed a course.

    Returns
    -------
    variants_df : list
        A list of data frames, where each data frame is a variant of the taggedRInput object.
    variants_co : list
        A list of lists of course offerings, where each list of course offerings is a variant of the taggedRInput object.
    '''
    
    # Check wether the taggedRInput object is valid or not.
    
    # Zero Check if the first column is called time
    if tagged_input.columns[0] != 'time':
        raise ValueError('The taggedRInput object does not have a first column called time.')
    else:
        # extract the time column.
        time = tagged_input['time']
        # remove the time column from the taggedRInput object
        tagged_input = tagged_input.drop(columns=['time'])
    
    
    # First Check Entries of pd dataframe are non_binary
    if len(np.unique(tagged_input)) <= 3:
        raise ValueError('The taggedRInput object has only ' +
                         str(len(np.unique(tagged_input))) +
                         ' unique element(s) and is likeli to be binary.  It must be non-binary.')
    
    # Second Check if items are rows and students are columns by checking the shape
    if tagged_input.shape[0] >= tagged_input.shape[1]:
        raise ValueError('The taggedRInput object is not in the correct shape.',  
                         'Items must be rows and students must be columns.')
    
    # Third Check if the columnnames are unique.
    if len(tagged_input.columns) != len(np.unique(tagged_input.columns)):
        raise ValueError('The taggedRInput object has duplicate column (students) names.')

    
    # Fourth Check if the columnnames and rownames are strings
    if not all(isinstance(x, str) for x in tagged_input.columns):
        raise ValueError('The taggedRInput object has non-string column (student) names.')
    if not all(isinstance(x, str) for x in tagged_input.index):
        raise ValueError('The taggedRInput object has non-string row (item) names.')
    
    # Fifth Check if students have more than one grade for rows with the same name
    
    # Create a list of unique row names
    unique_row_names = np.unique(tagged_input.index)

    for name in unique_row_names:
        # Locate the rows with the current row name
        rows = tagged_input.loc[name]
        #Check if course offering appears more than once
        if len(rows.shape)<2:
            continue

        # Check for each student if there are more than one grade different from -99999
        for stud_name in rows.columns:
            # Locate the grades of the current student
            grades = rows[stud_name]
            # Check if there are more than one grade different from -99999
            if len(np.unique(grades[grades != -99999])) > 1:
                raise ValueError('The taggedRInput object has more than one grade for the same student and item.')


    # Start processing
    variants = ['bin', 'non_bin', 'bin_red', 'non_bin_red']
    variants_df = []
    variants_co = []

    # Create a list of course offerings for each variant
    for variant in variants:
        co = []
        if 'red' in variant:
            # If the variant is reduced, then the course offerings are the 
            # the unique values of the column names.
            co.append(np.unique(tagged_input.index))
        else:
            # If the variant is not reduced, then the course offerings are the 
            # the column names + the element in the time array.
            for i,_ in enumerate(time):
                co.append(tagged_input.index[i] + ' ' + str(time[i]))
        variants_co.append(co)
        
    # Create a list of data frames for each variant
    for variant in variants:
        # Create a copy of the taggedRInput object
        df = tagged_input.copy()
        
        # If the variant is binary, then the grades are converted to binary
        if 'non_bin' not in variant:
            df[df.isin(fail_grades)] = 0
            df[df.isin(pass_grades)] = 1
        
        if 'red' in variant:
            # If the variant is reduced then build a new dataframe of ones,
            # where the rows are the unique values row names 
            # and the columns are the same as the original dataframe.
            df_red = pd.DataFrame(  data=np.ones((len(np.unique(tagged_input.index)),
                                                    tagged_input.shape[1])) * -99999,
                                    index=np.unique(tagged_input.index), 
                                    columns=tagged_input.columns)
            # Fill the new dataframe with the grades of the original dataframe
            # Go through each element of the original dataframe 
            # and fill the new dataframe with the grades of the original dataframe
            for i in range(df.shape[0]):
                for j in range(df.shape[1]):
                    #find out current row name
                    row_name = df.index[i]
                    #find out current column name
                    col_name = df.columns[j]
                    #set the grade of the new dataframe at the current row and column
                    #to the grade of the original dataframe at the current row and column
                    if df.iloc[i, j] == -99999:
                        continue
                    else:
                        df_red.at[row_name, col_name] = df.iloc[i, j]
            variants_df.append(df_red)
        else:
            variants_df.append(df)

    return variants_df, variants_co


def filter_repeated_courses(tagged_data, cos):
    '''
    This function taked a taggedRInput object and a list of course offerings and 
    returns a preprocessed data frame which includes only first try examinations.

    Parameters
    ----------
    tagged_data : taggedRInput object
        The taggedRInput object that is to be preprocessed.
    cos : list
        A list of reduced course offerings, meaning without time stamp.

    Returns
    -------
    tagged_data : non_binary version of the taggedRInput object
    '''
    # for each column (student) in the taggedRInput object get the list of all 
    # course offerings (index where value is not -99999)
    for col in tagged_data.columns:
        # get the list of all course offerings for the current student
        co = tagged_data[col][tagged_data[col] != -99999].index
        # get the list of all course offerings that are not in the reduced list
        co_to_remove = [x for x in co if x not in cos]
        # remove the course offerings that are not in the reduced list
        tagged_data[col][co_to_remove] = -99999

    return tagged_data


def df2group(tagged_data, item_times):
    '''
    This function takes a taggedRInput object and a list of course offering times and
    returns a StudentGroup object.

    Parameters
    ----------
    tagged_data : taggedRInput object
        The taggedRInput object that is to be preprocessed.
    item_times : list
        A list of course offering times.

    Returns
    -------
    group : StudentGroup object
        The StudentGroup object that was created from the taggedRInput object.
    '''
    # Construct StudentGroup object
    group = Classes.StudentGroup(name='group', students=[], members=[])
    
    # Replace the index of the tagged_data object with itsself and the time array

    course_names = tagged_data.index
    tagged_data.index = tagged_data.index + ' ' + item_times
    # For each column (student) in tagged_data create a Student object
    for col in tagged_data.columns:
        # Create a Student object
        student = Classes.Student(name=col, grades=[], courseNames=[], times=[])
        # For each row (course offering) in tagged_data
        for r, row in enumerate(tagged_data.index):
            #sprint(item_times[r], row, col, tagged_data[col][row])
            # If the grade is not -99999, then add the grade to the Student object
            if tagged_data[col][row] != -99999:
                student.setExam(grade= tagged_data[col][row], 
                                courseName=course_names[r],
                                time=item_times[r])
   
        #student.setDiscreteTimeValueVector()
        # Add the Student object to the StudentGroup object
        group.append(student)
    return group

def tests_on_tagged_data(tagged_data):
    '''
    This function tests if the elements in the time column are in the right format ('SSxy' or 'WSwx/yz')
    if not it raises an error.
    
    Parameters
    ----------
    tagged_data : taggedRInput object
        The taggedRInput object that is to be preprocessed.

    Returns
    -------
    None.
    '''  
    # Get the time column of the taggedRInput object
    time = tagged_data['time'].to_list()
    # For each element in the time column
    for t in time:
        # If the element is not in the right format, then raise an error
        if not re.match(r'^(SS|WS)\d{2}$', t) and not re.match(r'^(WS)\d{2}/\d{2}$', t):
            raise ValueError('The time column of the taggedRInput object is not in the right format.\
                             It should be in the format "SSxy" or "WSwx/yz" where w, x, y, z are digits.')
        
    return None

def taggedRInput_to_preprocessing(tagged_data, pass_grades, fail_grades, degree, verbose=False, cut_off_search=False, course_names=[]):
    '''
    This function takes a taggedRInput object and returns a list of preprocessed dataframes according to the variants and 
    the given grade types. This function is the alternative preprocessing method, if data is not given in raw format, but
    as a taggedRInput object istead. 

    Parameters
    ----------
    tagged_data : taggedRInput object
        The taggedRInput object that is to be preprocessed.
    pass_grades : list
        A list of grades that are considered as pass grades.
    fail_grades : list
        A list of grades that are considered as fail grades.
    degree : str
        The degree of the students in the taggedRInput object.
    verbose : bool, optional
        If True, the function prints out the number of students before and after preprocessing. The default is False.   

    Returns
    -------
    tagged_data : non_binary version of the taggedRInput object
    '''
    
    cwd = os.getcwd()
    # Construct the path to the data files
    data_folder = os.path.join(os.path.dirname(cwd),'data/real', degree)

    # Create an array of course offerings times 
    item_times = np.array(tagged_data['time'])
    for i, item in enumerate(item_times):
        if 'WS' in item:
            item_times[i] = item_times[i].replace('WS', 'W')

            # Find the index of / in the time
            slash_index = item_times[i].index('/')
            # Replace slash and everything after it with nothing
            item_times[i] = item_times[i][:slash_index]
        if 'SS' in item:
            item_times[i] = item_times[i].replace('SS', 'S')

    
    tests_on_tagged_data(tagged_data)

    # Delete the time column from the taggedRInput object
    tagged_data = tagged_data.drop(columns=['time'])

    if verbose:
        no_students = len(tagged_data.columns)
        print('pre_filter items', len(tagged_data.index))
    student_group = df2group(tagged_data, item_times)
    
    # Preprocess the student group
    student_group, pass_grades, fail_grades = StudentGroup_to_StudentGroup.group_filter(student_group,
                                                              pass_grades=pass_grades,
                                                              fail_grades=fail_grades,
                                                              verbose=verbose,
                                                              cut_off_search=cut_off_search,
                                                              course_names=course_names)
    # Dump the preprocessed student group into a pickle file in the data folder
    pickle.dump(student_group, open(os.path.join(data_folder, 'StudentGroup_filtered.pickle'), 'wb'))
    return pass_grades, fail_grades 

if __name__ == '__main__':

    # Specify number of students
    n_students = 6

    # Specify grades that are considered as fail grades
    fail_grades = np.arange(-5, 50)

    # Specify grades that are considered as pass grades
    pass_grades = np.arange(50, 105)
    
    # Write a dummy RInput file consisting of 10 items and 20 students with 2 courses each
    dummy = pd.DataFrame(np.random.randint(0,100,size=(n_students, 4)), columns=list('AACD'))
    dummy = dummy.transpose()

    # Give student rows in dummy random str names
    dummy.columns = [''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), 5)) for i in range(n_students)]
    
    # Randomly replace 1 of the grades of each student in the first two rows with -99999
    dummy.iloc[1,:] = -99999
    
    # Randomly replace 20% of the grades of each student in the last two rows with -99999
    dummy.iloc[0:,:] = dummy.iloc[0:,:].mask(np.random.random(dummy.iloc[0:,:].shape) < .2, -99999)

    #Insert a column at the first place of the dataframe specifying the time
    dummy.insert(0, 'time', ['1', '2', '1', '2'])
    


    print(dummy)
    #print(data_folder)


    #dfs, cos = taggedRInput_to_variants(dummy, fail_grades, pass_grades)

    #taggedRInput_to_preprocessing(dummy, fail_grades, pass_grades, verbose=True)

    #print all variants
    #for i, df in enumerate(dfs):
    #    print(df, '\n')
    #    print(cos[i], '\n')