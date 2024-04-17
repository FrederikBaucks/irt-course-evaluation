#This file preprocesses a StudentGroup object into a StudentGroup object.
#Students with less than 5 grades are removed.
#Courses with less than 10 grades are removed.
import sys
import os
import numpy as np
# Get the parent directory of the current file
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Add the parent directory to sys.path
sys.path.append(parent_dir)
# Import the module from the parental folder
import Classes
from collections import defaultdict
import cutoff_search
import importlib
importlib.reload(cutoff_search)


def filter_students(StudentGroup, fail_grades, direction):
    '''
    Goes through each student in a StudentGroup and removes students with less than 5 grades.

    Parameters
    ----------
    StudentGroup : StudentGroup
        The StudentGroup object to be filtered.
    fail_grades : list
        A list of grades that are considered failing. The min of this list is used as the threshold.
    direction : str
        The direction of grades: ascending or descending.
        
    Returns
    -------
    StudentGroup
        The filtered StudentGroup object.
    '''

    # Create a StudentGroup object to store the filtered students
    filtered_StudentGroup = Classes.StudentGroup(name=StudentGroup.name, 
                                                         students=[],
                                                         members=[])
    # Go through each student in the StudentGroup
    took_action = False
    for student in StudentGroup.students:

        # If the student has less than 5 nonzero grades, do not add them to the filtered StudentGroup
        
        if direction == 'ascending':
            thresh_grade = np.min(fail_grades)
            n_grades = student.grades[student.grades>thresh_grade]
            
        else:
            thresh_grade = np.max(fail_grades)
            n_grades = student.grades[student.grades<thresh_grade]
            
        if len(n_grades) < 5:
            took_action = True
            continue
        else:
            filtered_StudentGroup.append(student)
    return filtered_StudentGroup, took_action

def filter_courses(StudentGroup):
    '''
    Goes through each course in a StudentGroup and removes courses with less than 10 grades.

    Parameters
    ----------
    StudentGroup : StudentGroup
        The StudentGroup object to be filtered.

    Returns
    -------
    StudentGroup
        The filtered StudentGroup object.
    no_ids
        A list of course offering ids that were removed.
    '''
    
    #Find all course offering ids in the StudentGroup
    offering_ids = []
    offering_counts = []
    for student in StudentGroup.students:
        for c,course in enumerate(student.courseNames):
            offering_id = course + ' ' + student.times[c]
            if offering_id not in offering_ids:
                offering_ids.append(offering_id)
                offering_counts.append(1)
            else:
                offering_counts[offering_ids.index(offering_id)] += 1

# Default dict 
# from collections import defaultdict
# offering_count = defaultdict(lambda: 0)
# offering_counts[c] += 1
# offering_ids = list(offering_counts.keys())

    # Create a StudentGroup object to store only offerings with more than 10 grades
    filtered_StudentGroup = Classes.StudentGroup(name=StudentGroup.name,
                                                         students=[],
                                                         members=[])
    took_action = False
    for student in StudentGroup.students:
        # Create a Student object to store the filtered grades
        filtered_student = Classes.Student(name=student.name,
                                                   grades=[],
                                                   times=[],
                                                   courseNames=[])
        
        # Go through each grade in the student
        grades = []
        times = []
        courseNames = []
        for c,course in enumerate(student.courseNames):
            offering_id = course + ' ' + student.times[c]
            if offering_counts[offering_ids.index(offering_id)] < 20:
                took_action = True              
            else:
                grades.append(student.grades[c])
                times.append(student.times[c])
                courseNames.append(student.courseNames[c])

        
        if len(grades)>0:
            filtered_student.grades = np.array(grades)
            filtered_student.times = np.array(times)
            filtered_student.courseNames = np.array(courseNames)
            filtered_student.setDiscreteTimeValueVector()
            filtered_StudentGroup.append(filtered_student)
        else:
            #print('flag')
            took_action = True
    no_ids = len(offering_ids)
    return filtered_StudentGroup, no_ids, took_action

def filter_course_names(StudentGroup, course_names):
    '''
    Filter course_names from a StudentGroup object.

    Parameters
    ----------
    StudentGroup : StudentGroup
        The StudentGroup object to be filtered.
    course_names : list
        A list of course names to be included in the filtered StudentGroup.

    Returns
    -------
    StudentGroup
        The filtered StudentGroup object.
    '''
    # Create a StudentGroup object to store the filtered students
    filtered_StudentGroup = Classes.StudentGroup(name=StudentGroup.name,
                                                 students=[],
                                                 members=[])
    

    

    for student in StudentGroup.students:
            # Create a Student object to store the filtered grades
            filtered_student = Classes.Student(name=student.name,
                                                        grades=[],
                                                        times=[],
                                                        courseNames=[])
            for c,course in enumerate(student.courseNames):
                if course in course_names:
                    filtered_student.setExam(grade= student.grades[c],
                                             time= student.times[c],
                                             courseName= student.courseNames[c])
            if len(filtered_student.grades)>1:
                filtered_StudentGroup.append(filtered_student)
    return filtered_StudentGroup

def filter_repeated_courses(StudentGroup):
    '''
    Goes through each student in a StudentGroup and removes repeated courses.

    Parameters
    ----------
    StudentGroup : StudentGroup
        The StudentGroup object to be filtered.
    
    Returns
    -------
    StudentGroup
        The filtered StudentGroup object.
    '''

    # Create a StudentGroup object to store the filtered students
    filtered_StudentGroup = Classes.StudentGroup(name=StudentGroup.name,
                                                 students=[],
                                                 members=[])


    # Go through each student in the StudentGroup
    for student in StudentGroup.students:
        # Create a Student object to store the filtered grades
        filtered_student = Classes.Student(name=student.name,
                                                    grades=[],
                                                    times=[],
                                                    courseNames=[])
        for c,course in enumerate(student.courseNames):
            if 'WS' in student.times[c]:
                student.times[c] = student.times[c].replace('WS', 'W')
                # Find the index of / in the time
                slash_index = student.times[c].index('/')
                # Replace slash and everything after it with nothing
                student.times[c] = student.times[c][:slash_index]

            if 'SS' in student.times[c]:
                student.times[c] = student.times[c].replace('SS', 'S')
        for c,course in enumerate(student.courseNames):
            # find the index of all occurences of the course
            indices = [i for i, x in enumerate(student.courseNames) if x == course]
            # if the course occurs more than once
            if len(indices) > 1:
                student.setDiscreteTimeValueVector()
                # find the index of the first occurence of the course using the discrete time value vector
                first_index = indices[student.discreteTimes[indices].argmin()]
                # if the first occurence is the current occurence, add it to the filtered student
                if first_index == c:
                    filtered_student.setExam(grade= student.grades[c],
                                             time= student.times[c],
                                             courseName= student.courseNames[c])
            if len(indices) == 1:
                filtered_student.setExam(grade= student.grades[c],
                                         time= student.times[c],
                                         courseName= student.courseNames[c])
        filtered_student.setDiscreteTimeValueVector()        
        #print(len(student.courseNames))
        #print(len(filtered_student.courseNames))
        # add it to the filtered student
        filtered_StudentGroup.append(filtered_student)        
    return filtered_StudentGroup

def filter_courses_w_single_responses(StudentGroup, pass_grades):
    '''
    Goes through each course in a StudentGroup and removes courses when all students passed or failed.
    
    Parameters
    ----------
    StudentGroup : StudentGroup
        The StudentGroup object to be filtered.
    pass_grades : list
        A list of grades that are considered passing.

    Returns
    -------
    StudentGroup
        The filtered StudentGroup object.
    '''

    #Find all course offering ids in the StudentGroup
    offering_ids = []
    offering_counts = []
    offering_passes = []
    took_action = False
    for student in StudentGroup.students:
        for c,course in enumerate(student.courseNames):
            offering_id = course + ' ' + student.times[c]
            if offering_id not in offering_ids:
                offering_ids.append(offering_id)
                offering_counts.append(1)
                offering_passes.append(0)
            else:
                offering_counts[offering_ids.index(offering_id)] += 1

            if student.grades[c] <= np.max(pass_grades) and student.grades[c] >= np.min(pass_grades):
                offering_passes[offering_ids.index(offering_id)] += 1
    # Construct a list of course offering ids that should be removed
    remove_ids = []
    for i,offering_id in enumerate(offering_ids):
        # print(offering_id, offering_passes[i], offering_counts[i])
        
        if offering_passes[i] == offering_counts[i] or offering_passes[i] == 0:
            remove_ids.append(offering_id)
            took_action = True
    # Create a StudentGroup object to store the filtered students
  #  print(remove_ids, took_action)
    filtered_StudentGroup = Classes.StudentGroup(name=StudentGroup.name,
                                                    students=[],
                                                    members=[])
    # Go through each student in the StudentGroup
    for student in StudentGroup.students:
        # Create a Student object to store the filtered grades
        filtered_student = Classes.Student(name=student.name,
                                                    grades=[],
                                                    times=[],
                                                    courseNames=[])
        for c,course in enumerate(student.courseNames):
            offering_id = course + ' ' + student.times[c]
            if offering_id not in remove_ids:
                filtered_student.setExam(grade= student.grades[c],
                                         time= student.times[c],
                                         courseName= student.courseNames[c])
            
        # add it to the filtered student
        if len(filtered_student.grades) > 0:
            filtered_StudentGroup.append(filtered_student)
        else:
            #print('flag2')
            took_action = True

    unique_grades = defaultdict(dict)        
    for student in filtered_StudentGroup.students:
        for c,course in enumerate(student.courseNames):
            offering_id = course + ' ' + student.times[c]
            unique_grades[offering_id][student.grades[c] in pass_grades] = 1
    assert min([len(unique_grades[k]) for k in unique_grades]) == 2, "Need at least two grades"    
    
    return filtered_StudentGroup, took_action

def group_filter(StudentGroup, pass_grades, fail_grades, verbose=False, cut_off_search=False, course_names=[]):
    '''
    Filters a StudentGroup object by removing students with less than 5 grades 
    and courses with less than 10 grades until no more students or courses are removed.

    Parameters
    ----------
    StudentGroup : StudentGroup
        The StudentGroup object to be filtered.
    pass_grades : list
        A list of grades that are considered passing.
    fail_grades : list
        A list of grades that are considered failing.
    verbose : bool, optional
        If True, prints the number of students and courses before and after each filter.
        The default is False.
    cutoff_search : bool, optional
        If True, the function will return the number of students and courses after each filter.
        The default is False.

    Returns
    -------
    StudentGroup
        The filtered StudentGroup object.
    '''



    if np.max(pass_grades) < np.min(fail_grades):
        direction = 'descending'
    else:
        direction = 'ascending'

    if verbose:
        print('pre_filter students', len(StudentGroup.students))

    #Include only courses in course_names
    if len(course_names)>0:
       StudentGroup = filter_course_names(StudentGroup, course_names)

    


    # Inlcude only first occurences of repeated courses
    StudentGroup = filter_repeated_courses(StudentGroup)


    if cut_off_search:
        # Calculate the cut_off grade as the mean of the grade distribution
        if direction == 'ascending':
            avg_grades = [np.mean(s.grades) for s in StudentGroup.students if len(s.grades[s.grades>=np.min(pass_grades)])>=5]
        else:
            avg_grades = [np.mean(s.grades) for s in StudentGroup.students if len(s.grades[s.grades>=np.max(pass_grades)])>=5]
        # Delete nan values
        avg_grades = [x for x in avg_grades if str(x) != 'nan']
        
        cut_off, entropy = cutoff_search.best_split_continuous(np.array(avg_grades))

        #Concenate pass and fail grade
        grades = np.concatenate((pass_grades, fail_grades))
        # Check if grades are ascending or descending
        if np.min(pass_grades)>np.max(fail_grades):
            type = 'ascending'
            pass_grades = grades[grades>cut_off]
            fail_grades = grades[grades<=cut_off]
        else:
            type = 'descending'
            pass_grades = grades[grades<cut_off]
            fail_grades = grades[grades>=cut_off]
        print('Cut_off grade is set to: ', cut_off)

    no_action = False
    #total_grades = 0
    #total_grades_change = True
    while no_action==False:
        
            #total_grades_temp = np.sum([len(s.grades) for s in StudentGroup.students])
            #if verbose:
            #    print(total_grades, total_grades_temp)
            #if total_grades != total_grades_temp:
            #    total_grades = total_grades_temp
            #    total_grades_change = True
            #else:
            #    total_grades_change = False
            #    print(total_grades, total_grades_temp)
            
            # Filter students with less than 5 grades  
            c_names = np.unique([course for student in StudentGroup.students
                                        for course in student.courseNames])
            print('Current number of courses: ', len(c_names))

            StudentGroup, action_2 = filter_students(StudentGroup, fail_grades, direction)
            
            c_names = np.unique([course for student in StudentGroup.students
                                        for course in student.courseNames])
            print('Current number of courses after filtering students: ', len(c_names))

            StudentGroup, no_ids, action_3 = filter_courses(StudentGroup)
            StudentGroup, action_1 = filter_courses_w_single_responses(StudentGroup, pass_grades)

            c_names = np.unique([course for student in StudentGroup.students
                                        for course in student.courseNames])
            print('Current number of courses after filtering courses: ', len(c_names))
            if action_1:
                unique_grades = defaultdict(dict)

                for student in StudentGroup.students:
                    for c,course in enumerate(student.courseNames):
                        offering_id = course + ' ' + student.times[c]
                        unique_grades[offering_id][student.grades[c] in pass_grades] = 1
                assert min([len(unique_grades[k]) for k in unique_grades]) == 2, "Need at least two grades"
            #if action_2:
            #    test_action_2(StudentGroup, fail_grades, direction)
            #if action_3:
            #    test_action_3(StudentGroup)

        # print(action_1, action_2, action_3)
            if action_1 or action_2 or action_3:
                no_action = False
            else:
                no_action = True
        
    
    if verbose:
        print('post_filter students', len(StudentGroup.students))
        print('post_filter items', no_ids)
    return StudentGroup, pass_grades, fail_grades



    