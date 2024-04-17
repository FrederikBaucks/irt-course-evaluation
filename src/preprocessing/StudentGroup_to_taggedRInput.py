#This file preprocesses a StudentGroup object into a pandas data frame 
# suitable for MIRT.
import sys
import os
import numpy as np
import pandas as pd
# Get the parent directory of the current file
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Add the parent directory to sys.path
sys.path.append(parent_dir)
# Import the module from the parental folder
import Classes



def StudentGroup_to_df(StudentGroup, fail_grades, pass_grades, time_col=False):
    '''
    This function takes a StudentGroup object and returns a pandas data frame where 
    each row is a student and each column is a course offering. The entries are 0, 1, -99999,
    where 0 indicates that the student failed the course, 
    1 indicates that the student passed,
    and -99999 indicates that the student did not take the course.
    The data frame is suitable for MIRT.

    Parameters
    ----------
    StudentGroup : StudentGroup
        The StudentGroup object to be converted to a data frame.
    fail_grades : list
        A list of grades that indicate that a student failed a course.
    pass_grades : list
        A list of grades that indicate that a student passed a course.

    Returns
    -------
    df : pandas data frame
        The data frame where each row is a student and each column is a course offering.
        The entries are 0, 1, -99999, where 0 indicates that the student failed the course, 
        1 indicates that the student passed,
        and -99999 indicates that the student did not take the course.
    '''

    variants = ['bin', 'non_bin', 'bin_red', 'non_bin_red']
    variants_df = []
    variants_co = []
    for variant in variants:
        # Create a list of all the course offerings
        course_offerings = []
        for student in StudentGroup.students:
            for c,course in enumerate(student.courseNames):
                if 'red' in variant or time_col:
                    offering_id = course
                else:
                    offering_id = course + ' ' + student.times[c]
                if offering_id not in course_offerings:
                    course_offerings.append(offering_id)
                    
        # Create a data frame with the correct number of rows and columns
        df = pd.DataFrame(np.ones((len(StudentGroup.students), len(np.unique(course_offerings))))*-99999,
                            index=[student.name for student in StudentGroup.students],
                            columns=np.unique(course_offerings))  
        
        

        names = set([student.name for student in StudentGroup.students])
        assert len(names) == len(StudentGroup.students), "found duplicate names"

        # Fill in the data frame
        for student in StudentGroup.students:
            for c,course in enumerate(student.courseNames):
                if 'red' in variant:
                    offering_id = course
                else:
                    offering_id = course + ' ' + student.times[c]
                    assert student.name in df.index, 'student name not found ' + student.name
                    assert offering_id in df.columns, 'offering id not found ' + offering_id
                    #assert student.grades[c] in fail_grades or student.grades[c] in pass_grades, 'grade not found ' + str(student.grades[c])
                if 'non' in variant:
                    df.loc[student.name, offering_id] = student.grades[c]
                else: # x \in interval(0,1) \in \R 
                    if student.grades[c] >= np.min(fail_grades) and student.grades[c] <= np.max(fail_grades):
                        df.loc[student.name, offering_id] = 0
                    elif student.grades[c]  >= np.min(pass_grades) and student.grades[c] <= np.max(pass_grades):
                        df.loc[student.name, offering_id] = 1
                    else:
                        raise ValueError('grade not found ' + str(student.grades[c]))
        
        # Convert the data frame entries into integers
        #df = df.astype(float)

        # Transpose the data frame
        
        df = df.transpose()
        #print(df.columns[0])
        #first_col_name = df.columns[0]
        if not time_col:
            #df = df.set_index(df.columns[0])
            df.index = df[df.columns[0]]
             
        #print(df.columns[0])
        variants_df.append(df)
        variants_co.append(np.unique(course_offerings))
    return variants_df, variants_co

def StudentGroup2df_alt_test(StudentGroup):
    # Create a list of all the course offerings and a list of all times
    course_offerings = []
    times = []
    courses = []
    for student in StudentGroup.students:
        for c,course in enumerate(student.courseNames):
            offering_id = course + ' ' + student.times[c]
            if offering_id not in course_offerings:
                course_offerings.append(offering_id)
                times.append(student.times[c])
                courses.append(course)

    # Create data frame of ones, where each row is a course offering, but the rowname is just the corresponding course name
    # and each column is a student. In addition we add a column for the time as the first column.
    df = pd.DataFrame(np.ones((len(course_offerings), len(StudentGroup.students)))*-99999,
                        index=course_offerings,
                        columns=[student.name for student in StudentGroup.students])
    
    # Fill in the data frame
    for student in StudentGroup.students:
        for c,course in enumerate(student.courseNames):
            offering_id = course + ' ' + student.times[c]
            df.loc[offering_id, student.name] = student.grades[c]

    df.index = courses



    df.insert(0, 'time', times)



    return df