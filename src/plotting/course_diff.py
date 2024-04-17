import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import os

# Get parent directory of current file
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#Get parent dir of parent dir
parent_dir = os.path.dirname(parent_dir)



def get_reversed_sorting(degree):
    data_folder = parent_dir + '/data/real/' + degree + '/'

    # Load data
    course_names_ger = np.array(pd.read_csv(data_folder + 'course_names.csv', sep=',').sort_values(by=['Nr'])['Modul'])
    course_names_eng = np.array(pd.read_csv(data_folder + 'course_names.csv', sep=',').sort_values(by=['Nr'])['Modul'])
    items = np.array(pd.read_csv(data_folder + "binary/item_ids.csv", header=None)).flatten()
    



    item_time_df = []

    for i,_ in enumerate(items):
        if items[i].find('WS')!=-1:
            cur_course = items[i][:items[i].find('WS')]
            course_ind = np.where(course_names_ger==cur_course)[0]
            items[i] =  course_names_eng[course_ind][0] + ' ' + items[i][items[i].find('WS'):]
            item_time_df.append([course_names_eng[course_ind][0], items[i][items[i].find('WS'):]])
        if items[i].find('SS')!=-1:
            cur_course = items[i][:items[i].find('SS')]
            course_ind = np.where(course_names_ger==cur_course)[0]
            items[i] =  course_names_eng[course_ind][0] + ' ' + items[i][items[i].find('SS'):]
            item_time_df.append([course_names_eng[course_ind][0], items[i][items[i].find('SS'):]])
    item_time_df = pd.DataFrame(item_time_df)
    item_time_arr = np.array(item_time_df)
    temp_items = []
    for i in range(len(item_time_arr)):
        if item_time_arr[i,1].find('WS')!=-1:
            item_time_arr[i,1] =  item_time_arr[i,1][:item_time_arr[i,1].find('WS')] + item_time_arr[i,1][item_time_arr[i,1].find('WS')+2:]
        if item_time_arr[i,1].find('SS')!=-1:
            item_time_arr[i,1] =  item_time_arr[i,1][:item_time_arr[i,1].find('SS')] + item_time_arr[i,1][item_time_arr[i,1].find('SS')+2:]
        temp_items.append(item_time_arr[i,0] + ' ' + item_time_arr[i,1])

    reversed_sorting = np.argsort(np.array(temp_items))[::-1]
    return reversed_sorting

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable


def plot_course_offering_difficulty(degree, verbose=False, sort_courses='alphabetically', sort_times='alphabetically'):
    '''
    This function plots the course offering difficulty for a given degree. It sorts the course offerings by the sort parameter.
    
    Parameters
    ----------
    degree : str
        The degree for which the course offering difficulty should be plotted.
    verbose : bool, optional
        If True, the function prints additional information. The default is False.
    sort_courses : str, optional
        The sorting of the course offerings. The string can be one option out of ['alphabetically']. The default is 'alphabetically'.
    sort_times : str, optional
        The sorting of the times. The string can be one option out of ['alphabetically']. The default is 'alphabetically'.

    Returns
    -------
    None.
    '''

    # Load the data
    data_folder = parent_dir + '/data/real/' + degree + '/'

    course_names = np.array(pd.read_csv(data_folder + 'course_names.csv', 
                                        sep=',')['Modul'])
    
    item_difficulties = (-1)*np.array(pd.read_csv(data_folder + "irt_results/1pl_1dim/param_d_1PL_1DIM.csv"))
    #(-1)*np.array(pd.read_csv(data_folder + "irt_results/1pl_1dim/boot_d_Rasch.csv"))
    
    #item_difficulties = boot_item_difficulties.mean(axis=1)
    #item_stds = 2 * boot_item_difficulties.std(axis=1)
    item_stds = np.array(pd.read_csv(data_folder + "irt_results/1pl_1dim/ci_1PL_1DIM.csv"))
   

    items = np.array(pd.read_csv(data_folder + "binary/item_ids.csv", 
                                 header=None)).flatten()
    

    # Check if course_names appear in items
    if verbose:
        print('Checking if first two course names appear in items...')
        items_of_course = [x for x in items if course_names[0] + ' ' in x]
        if len(items_of_course) == 0:
            assert False, 'Course names do not appear in items. Check if course names are correct.'
        else:
            print('Course names appear in items.')


    # For each course in course_names, find the corresponding items 
    course_items = []
    # Create same list structure for item difficulties
    course_item_difficulties = []
    course_item_stds = []
    for i, course in enumerate(course_names):
        items_of_course = [x for x in items if course + ' ' in x]
        course_items.append(items_of_course)

        item_difficulties_of_course = [item_difficulties[np.where(items==x)][0] for x in items if course + ' ' in x]
        item_stds_of_course = [item_stds[np.where(items==x)][0] for x in items if course + ' ' in x]
        course_item_difficulties.append(item_difficulties_of_course)
        course_item_stds.append(item_stds_of_course)
    # Sort the course names
    if sort_courses == 'alphabetically':
        course_names_sorting = np.argsort(course_names)
        course_names = course_names[course_names_sorting]
        course_item_difficulties = [x for _,x in sorted(zip(course_names,course_item_difficulties))]
        course_item_stds = [x for _,x in sorted(zip(course_names,course_item_stds))]
        course_items = [x for _,x in sorted(zip(course_names,course_items))]

    # Sort the times
    if sort_times == 'alphabetically':
        for i, course in enumerate(course_names):
            course_sorting = np.argsort(course_items[i])
            course_items[i] = [x for _,x in sorted(zip(course_items[i],course_items[i]))]
            course_item_difficulties[i] = [x for _,x in sorted(zip(course_items[i],course_item_difficulties[i]))]
            course_item_stds[i] = [x for _,x in sorted(zip(course_items[i],course_item_stds[i]))]
    # Plot the course offering difficulty in two subplots next to each other 
     
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,(30/140*len(items))/2))

    # Specify white padding between the two plots
    fig.tight_layout(pad=35)

    # Delete padding at the top of the plot
    fig.subplots_adjust(top=0.9)




    # PLot the course offering difficulty
    #fig, ax1 = plt.subplots(figsize=(18,(30/100*len(items))/2))
    # Specify white padding around the plot
    #fig.tight_layout(pad=15.0)

    # Split the figure into two parts and plot the parts next to each other. 
    # The left part contains the first half of courses, the right part the second half.
    #divider = make_axes_locatable(ax1)
    #ax2 = divider.append_axes("right", pad=5.0, size = 2.5)

    # Declare sizes of the two plots to be equal
    #ax1.set_aspect('equal')
    #ax2.set_aspect('equal')
    
    # Force the aspect ratio of the right part to be equal to the left part
    #ax2.set_aspect(ax1.get_aspect())

    # Positions of ax1 and ax2
    #pos1 = ax1.get_position()
    #pos2 = [pos1.x0, pos1.y0,  pos1.width, pos1.height]
    #ax1.set_position(pos2)

    #pos_ax2 = ax2.get_position()
    #print(pos1, pos_ax2)

    #Set a shared title
    fig.suptitle('Course Offering Difficulty', fontsize=16, y=0.95, x=0.4)

    # Plot the left part

    ax1.set_xlabel('Item difficulty')
    # Create leabels for ax1 y-axis
    seperator = int(np.round(len(course_names)/2))
    left_labels = [] 
    right_labels = []
    for c, _ in enumerate(course_items):
        if c < seperator:
            left_labels.append(course_names[c])
        else:
            right_labels.append(course_names[c])
    
    # Create left offerings labels
    left_offerings = []
    for c, _ in enumerate(left_labels):
        left_offerings.append(course_items[c])

    # Calc number of items per course
    num_items_per_course = [len(x) for x in left_offerings]

    # Calc the course categories for the yticks for ax1
    cat_yticks = []
    cat_sep_yticks = []
    for i, _ in enumerate(num_items_per_course):
        if i == 0:
            cat_yticks.append(num_items_per_course[i]/2)
        else:
            cat_yticks.append(num_items_per_course[i]/2 + sum(num_items_per_course[:i]))
            cat_sep_yticks.append(sum(num_items_per_course[:i]))

    # Plot hlines hole width between course cat using cat_sep_yticks 
    for i, _ in enumerate(cat_sep_yticks):
        ax1.hlines(cat_sep_yticks[i]+0.5, -10 , 10,colors='black',linewidth=0.5)

    # Set the yticks as the offering names (left_offerings) over the categories
    labels = []
    for c,_ in enumerate(left_offerings):
        labels.extend(left_offerings[c])
    
    # Pop the course name from the labels, by finding a course name in the labels
    for c,_ in enumerate(labels):
        for course in course_names:
            if course in labels[c]:
                # Check if course name is at the beginning of the string
                if labels[c].find(course)==0:
                    if labels[c][len(course)] != ' ':
                        continue
                labels[c] = labels[c][labels[c].find(course)+len(course):]
                break
    # Add top and bottom padding to the labels
    labels.insert(0,'')
    labels.append('')

    # Set the yticks and yticklabels
    ax1.set_yticks(np.arange(len(labels)))
    ax1.set_yticklabels(labels, fontsize=8)

    # Plot hlines between each ytick of ax1
    for i, _ in enumerate(labels):
        if i == len(labels)-1:
            continue
        
        ax1.hlines(i+0.5, -10 , 10,colors='black',linewidth=0.5, alpha=0.5)

    # Get minimum and maximum item difficulty and round to integer
    min_diff = np.floor(np.min(item_difficulties))-1
    max_diff = np.ceil(np.max(item_difficulties))+1

    # Set the xticks and xticklabels
    ax1.set_xticks(np.arange(min_diff,max_diff,1))

    # Set the plot limits for ax1 according to difficulty
    ax1.set_xlim(min_diff-1,max_diff+1)


    # Plot the categories left to tick labels of the y-axis as free text
    for i, _ in enumerate(cat_yticks):
        ax1.text(-15, cat_yticks[i], left_labels[i], fontsize=12, ha='right', va='center')

    # Plot the item difficulties as scatter plot using course_item_difficulties
    diffs = []
    stds = []
    for i, _ in enumerate(course_item_difficulties):
        if i < seperator:
            diffs.extend(course_item_difficulties[i])
            stds.extend(course_item_stds[i])

    print(np.array(stds).shape, np.array(diffs).shape)
    ax1.scatter(diffs, np.arange(len(diffs))+1, s=10, c='black', marker='o', alpha=0.5)
    ax1.errorbar(np.array(diffs).flatten(), np.arange(len(diffs))+1, xerr=np.array(stds).flatten(), fmt='none', c='black', alpha=0.5)
    # Plot the right part
    ax2.set_xlabel('Item difficulty')
    # Create right offerings labels
    right_offerings = []
    for c, _ in enumerate(right_labels):
        right_offerings.append(course_items[c+seperator])

    # Calc number of items per course
    num_items_per_course = [len(x) for x in right_offerings]

    # Calc the course categories for the yticks for ax2
    cat_yticks = []
    cat_sep_yticks = []
    for i, _ in enumerate(num_items_per_course):
        if i == 0:
            cat_yticks.append(num_items_per_course[i]/2)
        else:
            cat_yticks.append(num_items_per_course[i]/2 + sum(num_items_per_course[:i]))
            cat_sep_yticks.append(sum(num_items_per_course[:i]))

    # Plot hlines hole width between course cat using cat_sep_yticks
    for i, _ in enumerate(cat_sep_yticks):
        ax2.hlines(cat_sep_yticks[i]+0.5, -10 , 10,colors='black',linewidth=0.5)

    # Set the yticks as the offering names (right_offerings) over the categories
    labels = []
    for c,_ in enumerate(right_offerings):
        labels.extend(right_offerings[c])

    # Pop the course name from the labels, by finding a course name in the labels
    for c,_ in enumerate(labels):
        for course in course_names:
            if course in labels[c]:
                # Check if course name is at the beginning of the string
                if labels[c].find(course)==0:
                    if labels[c][len(course)] != ' ':
                        continue
                labels[c] = labels[c][labels[c].find(course)+len(course):]
                break
    # Add top and bottom padding to the labels
    labels.insert(0,'')
    labels.append('')
    # Set the yticks and yticklabels
    ax2.set_yticks(np.arange(len(labels)))
    ax2.set_yticklabels(labels, fontsize=8)

    # Plot hlines between each ytick of ax2
    for i, _ in enumerate(labels):
        if i == len(labels)-1:
            continue
        ax2.hlines(i+0.5, -10 , 10,colors='black',linewidth=0.5, alpha=0.5)

    # Get minimum and maximum item difficulty and round to integer
    min_diff = np.floor(np.min(item_difficulties))-1
    max_diff = np.ceil(np.max(item_difficulties))+1

    # Set the xticks and xticklabels
    ax2.set_xticks(np.arange(min_diff,max_diff,1))

    # Set the plot limits for ax2 according to difficulty
    ax2.set_xlim(min_diff-1,max_diff+1)

    # Plot the categories left to tick labels of the y-axis as free text
    for i, _ in enumerate(cat_yticks):
        ax2.text(-15, cat_yticks[i], right_labels[i], fontsize=12, ha='right', va='center')

    # Plot the item difficulties as scatter plot using course_item_difficulties
    diffs = []
    stds = []
    for i, _ in enumerate(course_item_difficulties):
        if i >= seperator:
            diffs.extend(course_item_difficulties[i])
            stds.extend(course_item_stds[i])
    ax2.scatter(diffs, np.arange(len(diffs))+1, s=10, c='black', marker='o', alpha=0.5)
    ax2.errorbar(np.array(diffs).flatten(), np.arange(len(diffs))+1, xerr=np.array(stds).flatten(), fmt='none', c='black', alpha=0.5)
    plt.savefig(data_folder + 'plots/diff.pdf', format="pdf")
    plt.show()
    
    return None


if __name__ == '__main__':
    plot_course_offering_difficulty(degree='test', verbose=True)