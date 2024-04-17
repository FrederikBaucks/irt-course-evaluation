import numpy as np

def new_rep (vector):
#Function that gives an int as Semester instead of e.g. 'WS17/18' for the hole array. 
#This is used in the setDiscreteTimeValueVector function. 
    temp_vector=np.zeros(len(vector))
    for i in range(len(vector)):
        if vector[i].find('S')!= (-1):
            temp_vector[i] = float(vector[i][1:])
        elif vector[i].find('W')!= (-1):
            temp_vector[i] = float(vector[i][1:3])+0.5
        
    # Trasnform temp_vector to a vector with floats
    temp_vector = np.array(temp_vector, dtype=float)


    new_vector = np.zeros(len(vector))
    start = np.min(temp_vector)
    current=start
    counter = 0
    

    
    while current<=np.max(temp_vector):
        indexList = np.where(temp_vector==current)
        #print(current, np.max(vector))
        for i in range(len(indexList)):
            new_vector[indexList[i]]=counter
                
        counter += 1
        current += 0.5    
    return new_vector



class Student:
    def __init__(self, name, grades=[], times=[], discreteTimes=[], courseNames=[], workloads=[], 
                 time_rank=[], exam_prop=[]):
            self.name = name
            self.grades = np.array(grades)
            self.times = np.array(times)
            self.courseNames = np.array(courseNames)
            self.discreteTimes = np.array(discreteTimes)
            self.setDiscreteTimeValueVector()
            self.workloads = np.array(workloads)
            self.workloads = np.zeros(len(self.times))
            self.time_rank = np.zeros(len(self.discreteTimes))
            self.exam_prop = np.zeros(len(self.times))
            
    def pop(self, index):
        self.grades=np.delete(self.grades, index)
        self.times=np.delete(self.times, index)
        self.courseNames=np.delete(self.courseNames, index)
        self.discreteTimes=np.delete(self.discreteTimes, index)
        return self.grades, self.times, self.courseNames, self.discreteTimes

    def pop_without_discreteTimes(self, index):
        self.grades=np.delete(self.grades, index)
        self.times=np.delete(self.times, index)
        self.courseNames=np.delete(self.courseNames, index)
        return self.grades, self.times, self.courseNames

    def zero_mean_grades(self):
        gpa = np.mean(self.grades)
        self.grades = self.grades-gpa
        return self.grades 

    def setGrade(self, value):
        self.grades = np.append(self.grades,value)
        return self.grades
    
    def setExam_prop (self, value):
        self.exam_prop = np.append(self.exam_prop,value)
        return self.exam_prop
    
    def setTime(self, value):
        self.times = np.append(self.times,value)
        return self.times
    
    def setDiscreteTimeValueVector(self):
        if(len(self.times) > 0):
            self.discreteTimes = new_rep(self.times)
        else:
            self.discreteTimes = []
        return self.discreteTimes    
    
    def setExam(self, grade, time, courseName):
        if len(self.grades)!=len(self.times):
            print('Warning: grades and times vectors have different length!', len(self.grades), 'vs.', len(self.times))
        if len(self.grades)!=len(self.courseNames):
            print('Warning: grades and times vectors have different length!', len(self.grades), 'vs.', len(self.courseNames))
        
        self.grades = np.append(self.grades,grade)
        self.times = np.append(self.times,time)
        self.courseNames = np.append(self.courseNames, courseName)
        return self.grades, self.times, self.courseNames
    def setWorkload(self, value):
        self.workloads = np.append(self.workloads,value)
    def getGrades(self):
        print(self.grades)
        return(self.grades)

class StudentGroup:
    def __init__(self, name, members=[], students=[]):
            self.name = name
            self.students = students
            self.members = members
    
    def append(self, NewStudent):
        self.students.append(NewStudent)
        self.members.append(NewStudent.name)
        return self.students, self.members
		
    def get_mean_grade(self):
        means = []
        for student in self.students:
            if len(student.grades) <5 :
                continue
            else:
                means.append(np.mean(student.grades))
        mean_grade = np.mean(means)
        return mean_grade
    
    def get_median_grade(self):
        medians = []
        for student in self.students:
            if len(student.grades) <5 :
                continue
            else:
                medians.append(np.median(student.grades))
        median = np.median(medians)
        return median
