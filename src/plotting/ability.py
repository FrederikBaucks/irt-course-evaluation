import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_ability(best_overall_model, data_folder, degree_names):
    colors = ['red', 'blue', 'violet']
    #print(data_folder)
    students_1 = np.array(pd.read_csv(data_folder + '/../' + degree_names[0] + '/binary_reduced/student_ids.csv', header=None)).flatten()
    if degree_names[1]!=None:
        students_2 = np.array(pd.read_csv(data_folder + '/../' + degree_names[1] + '/binary_reduced/student_ids.csv', header=None)).flatten()
        students_concat =  np.array(pd.read_csv(data_folder + '/../' + degree_names[0] + '_'+degree_names[1] + '/binary_reduced/student_ids.csv', header=None)).flatten()
    else:
        students_2 = students_1



    #print(len(students_1), len(students_2), len(students_concat))
    student_names = students_concat

    students_1 = list(set(students_1) & set(student_names))
    students_2 = list(set(students_2) & set(student_names))
    intersection = list(set(students_1) & set(students_2))


    # change the names to indices of the student_names array
    students_1 = [np.where(student_names == student)[0][0] for student in students_1]
    students_2 = [np.where(student_names == student)[0][0] for student in students_2]
    intersection = [np.where(student_names == student)[0][0] for student in intersection]
    if len(intersection) > 0:
        print('students enrolled in both majors: ' + str(len(intersection)))

    student_split = [list(set(students_1) - set(intersection)), list(set(students_2) - set(intersection)), intersection]

    

    # map mmod1_dim1 to 1pl_1dim, mmod2_dim1 to 2pl_1dim, mmod2_dim2 to 2pl_2dim, mmod2_dim3 to 2pl_3dim
    model_mapping = {'mmod1': ['1pl_1dim', 1], 'mmod2_dim1': ['2pl_1dim',1], 'mmod2_dim2': ['2pl_2dim',2], 'mmod2_dim3': ['2pl_3dim',3]}
    model = model_mapping[best_overall_model][0]
    dim = model_mapping[best_overall_model][1]

    # load difficulty values of best overall model
    abilities = pd.read_csv(data_folder + '/irt_results/' + model + '/abilities.csv')
    

    if dim == 2: 

        #plot abilities of students in both majors
        plt.figure(figsize=(8.5,5))
        colors = []
        for s, abi in enumerate(abilities.values):
            if s in intersection:
                colors.append('violet')
            elif s in student_split[0]:
                colors.append('red')
            elif s in student_split[1]:
                colors.append('blue')
            else:
                colors.append('gray')
        plt.scatter(abilities.iloc[:,0], abilities.iloc[:,1], color=colors, alpha=0.3)    
        plt.xlabel('ability dim 1')
        plt.ylabel('ability dim 2')
        plt.title("Ability Plot")
        plt.scatter([], [], color='red', label=degree_names[0], alpha=0.3)
        if degree_names[1]!=None:
            plt.scatter([], [], color='blue', label=degree_names[1], alpha=0.3)
        plt.legend()
        plt.show()

    if dim == 3: 
        #plot abilities of students in both majors
        plt.figure(figsize=(8.5,5))
        colors = []
        for s, abi in enumerate(abilities.values):
            if s in intersection:
                colors.append('violet')
            elif s in student_split[0]:
                colors.append('red')
            elif s in student_split[1]:
                colors.append('blue')
            else:
                colors.append('gray')
        plt.scatter(abilities.iloc[:,0], abilities.iloc[:,1], color=colors, alpha=0.3)    
        plt.xlabel('ability dim 1')
        plt.ylabel('ability dim 2')
        plt.title("Ability Plot")
        plt.scatter([], [], color='red', label=degree_names[0], alpha=0.3)
        if degree_names[1]!=None:
            plt.scatter([], [], color='blue', label=degree_names[1], alpha=0.3)
        plt.legend()
        plt.show()

        #plot abilities of students in both majors
        plt.figure(figsize=(8.5,5))
        colors = []
        for s, abi in enumerate(abilities.values):
            if s in intersection:
                colors.append('violet')
            elif s in student_split[0]:
                colors.append('red')
            elif s in student_split[1]:
                colors.append('blue')
            else:
                colors.append('gray')
        plt.scatter(abilities.iloc[:,0], abilities.iloc[:,2], color=colors, alpha=0.3)    
        plt.xlabel('ability dim 1')
        plt.ylabel('ability dim 3')
        plt.title("Ability Plot")
        plt.scatter([], [], color='red', label=degree_names[0], alpha=0.3)
        if degree_names[1]!=None:
            plt.scatter([], [], color='blue', label=degree_names[1], alpha=0.3)
        plt.legend()
        plt.show()

        #plot abilities of students in both majors
        plt.figure(figsize=(8.5,5))
        colors = []
        for s, abi in enumerate(abilities.values):
            if s in intersection:
                colors.append('violet')
            elif s in student_split[0]:
                colors.append('red')
            elif s in student_split[1]:
                colors.append('blue')
            else:
                colors.append('gray')
        plt.scatter(abilities.iloc[:,1], abilities.iloc[:,2], color=colors, alpha=0.3)    
        plt.xlabel('ability dim 2')
        plt.ylabel('ability dim 3')
        plt.title("Ability Plot")
        plt.scatter([], [], color='red', label=degree_names[0], alpha=0.3)
        if degree_names[1]!=None:
            plt.scatter([], [], color='blue', label=degree_names[1], alpha=0.3)
        plt.legend()
        plt.show()

        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot
        ax.scatter(abilities.iloc[:, 0], abilities.iloc[:, 1], abilities.iloc[:, 2], color=colors, alpha=0.3)
        ax.scatter([], [], [], color='red', label=degree_names[0], alpha=0.3)
        if degree_names[1]!=None:
            ax.scatter([], [], [], color='blue', label=degree_names[1], alpha=0.3)


        # Axes labels
        ax.set_xlabel('Abilities dim 1')
        ax.set_ylabel('Abilities dim 2')
        ax.set_zlabel('Abilities dim 3')

        
        ax.legend()
        
        # Title
        plt.title("3D Difficulty Plot")

    # load course names 
    course_names_list = pd.read_csv(data_folder + '/binary_reduced/item_ids.csv', header=None).values.flatten()
    # set course names as index

