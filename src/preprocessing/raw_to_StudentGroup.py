import os
import csv
import pandas as pd
import numpy as np

from .. import Classes

# Get the path to the data folder
data_folder = os.path.join(os.path.dirname(__file__), '..', '..', 'data')


# ask for an input specifying the degree
print('Which degree?')

# read folder names in data/real and print them, but ignore .gitkeep:
for i, degree in enumerate(os.listdir(os.path.join(data_folder, 'real'))):
    if degree != '.gitkeep':
        print(i+1, degree)

print('Enter the number of the degree you want to use:')
degree = input()

# overwrite degree with the name of the folder
degree = os.listdir(os.path.join(data_folder, 'real'))[int(degree)-1]

# Construct the path to the dataset file
dataset_path = os.path.join(data_folder, 'real', degree, 'raw.csv')

# Check the seperation in the csv file at dataset_path and set sep accordingly
with open(dataset_path, 'r') as csvfile:    
    dialect = csv.Sniffer().sniff(csvfile.read(10))
    csvfile.seek(0)
    reader = csv.reader(csvfile, dialect)
    sep = dialect.delimiter
    
# Read the dataset file
dataset = pd.read_csv(dataset_path, sep=sep, low_memory=False)

# go through each row of the dataset and create a Student object for each student
# create a list of all students
students = []
student_names = []
for i in range(len(dataset)):
    # get the name of the student
    name = dataset.iloc[i]['ID']
    if name not in student_names:
        student_names.append(name)
        
        # create a student object
        student = Classes.Student(name=name, 
                                          grades=[], 
                                          times=[], 
                                          workloads=[], 
                                          time_rank=[], 
                                          exam_prop=[])

    


