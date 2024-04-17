import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
MISSING_VALUE = -99999
import matplotlib.colors as mcolors

def plot_difficulty(best_overall_model, data_folder, degree_names, color_gradient = False):
    start_time = time.time()
    colors = ['red', 'blue']
    if color_gradient:
        # Define custom colormap
        custom_colors = [(0, "darkgreen"), (0.5, "yellow"), (1, "darkorange")]  # format: (position, color)
        n_bins = 100  # Number of bins
        # Create colormap
        cmap_name = "custom_cmap"
        colormap = mcolors.LinearSegmentedColormap.from_list(cmap_name, custom_colors, N=n_bins)

    # read the data from the csv file
    print(data_folder)
    tagged_data_1 = pd.read_csv(data_folder + '/../' + degree_names[0] + '/taggedRInput.csv', index_col=0)
    if degree_names[1]!=None:
        tagged_data_2 = pd.read_csv(data_folder + '/../' + degree_names[1] + '/taggedRInput.csv', index_col=0)
    else:
        tagged_data_2 = tagged_data_1
    concat_tagged_data = pd.read_csv(data_folder + '/taggedRInput.csv', index_col=0)
    # courses_1 = tagged_data_1.index.values
    # courses_2 = tagged_data_2.index.values
    course_names = concat_tagged_data.index.values
    # courses_1 = list(set(courses_1) & set(course_names))
    # courses_2 = list(set(courses_2) & set(course_names))
    # intersection = list(set(courses_1) & set(courses_2))


    courses_1 = tagged_data_1.index[tagged_data_1.index.isin(course_names)]
    courses_2 = tagged_data_2.index[tagged_data_2.index.isin(course_names)]
    intersection = courses_1[courses_1.isin(courses_2)]

    # # count columns where course values are not MISSING_VALUE
    # count_inter_1 = np.sum(tagged_data_1.loc[intersection] != MISSING_VALUE, axis=1)
    # count_inter_2 = np.sum(tagged_data_2.loc[intersection] != MISSING_VALUE, axis=1)
    count_inter_1 = (tagged_data_1.loc[intersection] != MISSING_VALUE).sum(axis=1)
    count_inter_2 = (tagged_data_2.loc[intersection] != MISSING_VALUE).sum(axis=1)

    
    inter_ratios = count_inter_1 / (count_inter_1 + count_inter_2)

    
    course_split = [list(set(courses_1) - set(intersection)), list(set(courses_2) - set(intersection)), intersection]
    # map mmod1_dim1 to 1pl_1dim, mmod2_dim1 to 2pl_1dim, mmod2_dim2 to 2pl_2dim, mmod2_dim3 to 2pl_3dim
    model_mapping = {'mmod1': ['1pl_1dim', 1], 'mmod2_dim1': ['2pl_1dim',1], 'mmod2_dim2': ['2pl_2dim',2], 'mmod2_dim3': ['2pl_3dim',3]}
    model = model_mapping[best_overall_model][0]
    dim = model_mapping[best_overall_model][1]

    # load difficulty values of best overall model
    difficulty = pd.read_csv(data_folder + '/irt_results/' + model + '/diff.csv')

    # load course names 
    course_names_list = pd.read_csv(data_folder + '/binary_reduced/item_ids.csv', header=None).values.flatten()
    # set course names as index
    difficulty.index = course_names_list

    print('building data', time.time() - start_time)
    # plot difficulty values according to dim in scatter plot

    # if degree_names[1] is not None:
    #     fig = plt.figure(figsize=(2, 5))  # Adjust the size as needed

    #     # Define your colormap and the mappable object
    #     sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=0, vmax=1))
    #     sm._A = []

    #     # Add colorbar to the figure
    #     cbar = fig.colorbar(sm, label='intersection ratio', orientation='vertical', aspect=10)
    #     cbar.set_label('intersection ratio', rotation=270, labelpad=20)

    #     plt.axis('off')  # Turn off the axis
    #     plt.show()
    plt.figure(figsize=(8.5,5))

    if dim == 1:
        # plt.scatter(difficulty, np.zeros(len(difficulty)), color='black', alpha=0.7)
        # plt.xlabel('difficulty')
        # plt.ylabel('dim')
        # plt.title("Difficulty Plot")
        # for i, txt in enumerate(course_names_list):
        #     plt.annotate(txt[:5], (difficulty.iloc[i, 0], 0), textcoords="offset points", xytext=(0,5), ha='center', rotation=90)
        print(difficulty.sort_values(by='x'))
    elif dim == 2:
        start_time = time.time()
        # get row index of courses with difficulty abs()>10 in some column
        idx = difficulty[(abs(difficulty.iloc[:,0])>20) | (abs(difficulty.iloc[:,1])>20)].index
        # Warn about courses with difficulty abs()>10 in some column and show them with values
        print("Warning: The following courses have difficulty values with abs()>10 in some column:")
        print(difficulty.loc[idx])
        print("These courses will be removed from the plot.")

        # delete the row from difficulty
        difficulty = difficulty.drop(idx)

        plt.scatter(difficulty.iloc[:,0], difficulty.iloc[:,1], color='black', alpha=0.7)
        plt.xlabel('difficulty dim 1')
        plt.ylabel('difficulty dim 2')
        plt.title("Difficulty Plot")



        # Extracting only uppercase letters and digits from each index
        txt_copies = difficulty.index.str.replace(r'[^A-Z0-9]', '', regex=True)

        #print(difficulty.index)
        # Prepare conditions for numpy.select
        conditions = [
            difficulty.index.isin(course_split[0]),
            difficulty.index.isin(course_split[1]),
            # Additional conditions for color_gradient and inter_ratios
        ]

        # Prepare choice of colors for each condition
        choices = [colors[0], colors[1], 
                # Add the corresponding colors for color_gradient and inter_ratios
                ]

        # Default color if no conditions are met
        default_color = 'black'

        # Apply vectorized color assignment
        color_array = np.select(conditions, choices, default=default_color)

        for i, (txt, color) in enumerate(zip(txt_copies, color_array)):
            plt.scatter(difficulty.iloc[i, 0], difficulty.iloc[i, 1], color=color, alpha=0.9)
            plt.annotate(txt, (difficulty.iloc[i, 0], difficulty.iloc[i, 1]), textcoords="offset points", xytext=(0,5), ha='center', color='black')


        # for i, txt in enumerate(difficulty.index):
        #     # delete all non capital letters from txt and all non numeric characters from txt
        #     txt_copy = ''.join(c for c in txt if c.isupper() or c.isdigit())
        #     if txt in course_split[0]:
        #         temp_col = colors[0]
        #     elif txt in course_split[1]:
        #         temp_col = colors[1]
        #     else:
        #         if color_gradient:
        #             # find txt in index of inter_ratios
        #             idx = inter_ratios.index[inter_ratios.index==txt]
        #             # get the value of inter_ratios at index idx
        #             temp_val = inter_ratios[idx].values[0]
        #             #
        #             #print('inter', inter_ratios)
        #             temp_col = colormap(temp_val)
        #         else:
        #             idx = inter_ratios.index[inter_ratios.index==txt]
        #             if inter_ratios[idx].values[0] < 0.5:
        #                 #print(inter_ratios)
        #                 temp_col = colors[1]
        #             else:
        #                 temp_col = colors[0]
        #     plt.scatter(difficulty.iloc[i,0], difficulty.iloc[i,1], color=temp_col, alpha=0.9)
        #     plt.annotate(txt_copy, (difficulty.iloc[i, 0], difficulty.iloc[i, 1]), textcoords="offset points", xytext=(0,5), ha='center', color='black')
        for color, label in zip(colors, degree_names):
            plt.scatter([], [], color=color, label=label)
            plt.legend()
        # plot colorbar for intersection beside the plot
        # if degree_names[1]!=None:
        #     sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=0, vmax=1))
        #     sm._A = []
        #     plt.colorbar(sm, label='intersection ratio')

        print('2 dim plot', time.time()-start_time)
    elif dim == 3:
        start_time = time.time()
        # plt.scatter(difficulty.iloc[:,0], difficulty.iloc[:,1], color='black', alpha=0.7)
        # plt.xlabel('difficulty dim 1')
        # plt.ylabel('difficulty dim 2')
        # plt.title("Difficulty Plot")
        # plt.figure(figsize=(8.5,5))
        # plt.scatter(difficulty.iloc[:,0], difficulty.iloc[:,2], color='black', alpha=0.7)
        # plt.xlabel('difficulty dim 1')
        # plt.ylabel('difficulty dim 3')
        # plt.title("Difficulty Plot")
        # plt.figure(figsize=(8.5,5))
        # plt.scatter(difficulty.iloc[:,1], difficulty.iloc[:,2], color='black', alpha=0.7)
        # plt.xlabel('difficulty dim 2')
        # plt.ylabel('difficulty dim 3')
        # plt.title("Difficulty Plot")

        # get row index of courses with difficulty abs()>10 in some column
        idx = difficulty[(abs(difficulty.iloc[:,0])>20) | (abs(difficulty.iloc[:,1])>20) | (abs(difficulty.iloc[:,2])>20)].index
        # Warn about courses with difficulty abs()>10 in some column and show them with values
        print("Warning: The following courses have difficulty values with abs()>10 in some column:")
        print(difficulty.loc[idx])
        print("These courses will be removed from the plot.")

        # delete the row from difficulty
        difficulty = difficulty.drop(idx)

        color_list=[]
        for i, txt in enumerate(difficulty.index):
            # delete all non capital letters from txt and all non numeric characters from txt
            txt_copy = ''.join(c for c in txt if c.isupper() or c.isdigit())
            if txt in course_split[0]:
                color_list.append(colors[0])
            elif txt in course_split[1]:
                color_list.append(colors[1])
            else:
                if color_gradient:
                    # find txt in index of inter_ratios
                    idx = inter_ratios.index[inter_ratios.index==txt]
                    # get the value of inter_ratios at index idx
                    temp_val = inter_ratios[idx].values[0]

                    #
                    temp_col = colormap(temp_val)
                else:
                    idx = inter_ratios.index[inter_ratios.index==txt]
                    if inter_ratios[idx].values[0] < 0.5:
                        temp_col = colors[1]
                    else:
                        temp_col = colors[0]
                color_list.append(temp_col)
        

        # First scatter plot
        plt.figure(figsize=(8.5,5))
        plt.scatter(difficulty.iloc[:,0], difficulty.iloc[:,1], color=color_list, alpha=0.9)
        plt.xlabel('difficulty dim 1')
        plt.ylabel('difficulty dim 2')
        plt.title("Difficulty Plot")
        for i, txt in enumerate(difficulty.index):
            txt_copy = ''.join(c for c in txt if c.isupper() or c.isdigit())
            plt.annotate(txt_copy, (difficulty.iloc[i, 0], difficulty.iloc[i, 1]), textcoords="offset points", xytext=(0,5), ha='center')
        for color, label in zip(colors, degree_names):
            plt.scatter([], [], color=color, label=label)
            plt.legend()
        # if degree_names[1]!=None:
        #     sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=0, vmax=1))
        #     sm._A = []
        #     plt.colorbar(sm, label='intersection ratio')
        # Second scatter plot
        plt.figure(figsize=(8.5,5))
        plt.scatter(difficulty.iloc[:,0], difficulty.iloc[:,2], color=color_list, alpha=0.7)
        plt.xlabel('difficulty dim 1')
        plt.ylabel('difficulty dim 3')
        plt.title("Difficulty Plot")
        for i, txt in enumerate(difficulty.index):
            txt_copy = ''.join(c for c in txt if c.isupper() or c.isdigit())
            plt.annotate(txt_copy, (difficulty.iloc[i, 0], difficulty.iloc[i, 2]), textcoords="offset points", xytext=(0,5), ha='center')
        for color, label in zip(colors, degree_names):
            plt.scatter([], [], color=color, label=label)
            plt.legend()
        # if degree_names[1]!=None:
        #     sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=0, vmax=1))
        #     sm._A = []
        #     plt.colorbar(sm, label='intersection ratio')

        # Third scatter plot
        plt.figure(figsize=(8.5,5))
        plt.scatter(difficulty.iloc[:,1], difficulty.iloc[:,2], color=color_list, alpha=0.7)
        plt.xlabel('difficulty dim 2')
        plt.ylabel('difficulty dim 3')
        plt.title("Difficulty Plot")
        for i, txt in enumerate(difficulty.index):
            txt_copy = ''.join(c for c in txt if c.isupper() or c.isdigit())
            plt.annotate(txt_copy, (difficulty.iloc[i, 1], difficulty.iloc[i, 2]), textcoords="offset points", xytext=(0,5), ha='center')

        for color, label in zip(colors, degree_names):
            plt.scatter([], [], color=color, label=label)
        # if degree_names[1]!=None:
        #     sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=0, vmax=1))
        #     sm._A = []
        #     plt.colorbar(sm, label='intersection ratio')
        
        
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot
        ax.scatter(difficulty.iloc[:, 0], difficulty.iloc[:, 1], difficulty.iloc[:, 2], color=color_list, alpha=0.7)

        # Adding labels
        for i, txt in enumerate(difficulty.index):
            txt_copy = ''.join(c for c in txt if c.isupper() or c.isdigit())
            ax.text(difficulty.iloc[i, 0], difficulty.iloc[i, 1], difficulty.iloc[i, 2], txt_copy, size=10, ha="center")

        # Axes labels
        ax.set_xlabel('Difficulty dim 1')
        ax.set_ylabel('Difficulty dim 2')
        ax.set_zlabel('Difficulty dim 3')

        for color, label in zip(colors, degree_names):
            ax.scatter([], [], [], color=color, label=label)
        
        # if degree_names[1]!=None:
        #     sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=0, vmax=1))
        #     sm._A = []
        #     plt.colorbar(sm, label='intersection ratio')
        
        ax.legend()
        
        # Title
        plt.title("3D Difficulty Plot")
        print('dim 3 plotting ', time.time()-start_time)

    plt.show()

   
    return difficulty.sort_values(by=difficulty.columns[0])