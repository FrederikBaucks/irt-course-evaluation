import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# 3d plot with group ratio and group offset and error
from mpl_toolkits.mplot3d import Axes3D

from scipy.interpolate import griddata
def simulate_no_courses(no_courses, no_students, no_simulations, group_size = [x/500 for x in range(1, 70, 10)], group_offset = [(x, y) for x in np.arange(-2, 0, 1) for y in np.arange(1, 3, 1)], interpolation='linear', verbose=False, offset= True):

    group_ratio = 0.5

    # DCF course 
    dcf = 0

    
    # Group offsets
    

    #sim_err = []
    global_sim_err = []
    global_err = []
    global_intercepts = []
    power_offset = []
    for g_size in group_size:
        errors = []
        intercepts = []
        sim_err = []
        power_list = []
        for offset in group_offset:
            power = 0
            current_sim_err = []
            for sim in range(no_simulations):
                # split students into two groups based on group ratio
                #no_students = 2*g_size #+ (1 - group_ratio) * g_size
                #no_students = int(no_students)
                # Generate IRT parameters
                theta = np.random.normal(0, 1, no_students)
                d = np.random.normal(0, 1, no_courses)
                d = d.reshape((no_courses, 1))

                # Generate student responses
                p = 1 / (1 + np.exp(-(theta - d)))
                y = np.random.binomial(1, p)
                group1 = np.random.choice(np.arange(0, no_students), int(g_size), replace=False)#no_students*group_ratio)
                group2 = np.setdiff1d(np.arange(0, no_students), group1)

                y_new = y.copy()

                # calculate new responses for the dcf course
                p_1 = 1 / (1 + np.exp(-(theta[group1] - d[dcf] + offset[0])))
                p_2 = 1 / (1 + np.exp(-(theta[group2] - d[dcf] + offset[1])))

                y_new[dcf, group1] = np.random.binomial(1, p_1)
                y_new[dcf, group2] = np.random.binomial(1, p_2)

                # fit the logistic regression model for dcf
                from statsmodels.formula.api import glm
                import statsmodels.api as sm
                group_name = [-1 if x in group1 else 1 for x in range(0, no_students)]
                difficulty = [d[dcf][0] for x in range(0, no_students)]
                data = pd.DataFrame({'y': y_new[dcf], 'theta': theta, 'group': group_name, 'difficulty': difficulty})

                reg_form = 'y ~ group + 1'
                if offset:
                    offset_data = data['theta'] - data['difficulty'] 
                else:
                    offset_data = np.zeros(no_students)
                model = glm(reg_form, data=data, family=sm.families.Binomial(), offset=offset_data)
                # get the p value of the group coefficient
                p_value = model.fit().pvalues[1]
     
                if p_value < 0.05:
                    power += 1
                intercept = model.fit().params[0]
                group = model.fit().params[1]
                p_pred_1 = 1 / (1 + np.exp(-(theta[group1] - d[dcf][0] + intercept - group))) 
                p_pred_2 = 1 / (1 + np.exp(-(theta[group2] - d[dcf][0] + intercept + group)))   
                err = np.mean(np.mean(np.abs(p_pred_1 - p_1)) + np.mean(np.abs(p_pred_2 - p_2)))
                

                
                current_sim_err.append(err)
            power = power / no_simulations
            power_list.append(power)
            errors.append(err)
            intercepts.append(intercept)
                # if verbose:
            #     plt.figure(figsize=(5, 2))
            #     plt.scatter(theta[group2], p_2, alpha=1, marker='.', s=2, label= '1')
            #     plt.scatter(theta[group2], p_pred_2, alpha=1, marker='.', s=2, label= '1 recovered')
            #     plt.scatter(theta, p[dcf], alpha=1, marker='.', s=1, label='rasch')
            #     plt.scatter(theta[group1], p_1, alpha=1, marker='.', s=2, label = '-1')
            #     plt.scatter(theta[group1], p_pred_1, alpha=1, marker='.', s=2, label = '-1 recovered')
            #     plt.xlabel('Theta')
            #     plt.ylabel('Proportion Correct')
            #     plt.title('Error: ' + str(round(err, 2)))
            #     # set font size
            #     plt.rcParams.update({'font.size': 10})
            #     plt.legend()
            #     plt.show()
            sim_err.append((np.mean(current_sim_err), np.std(current_sim_err)))
        power_offset.append(power_list)
        global_err.append(errors)
        global_intercepts.append(intercepts)
        global_sim_err.append(sim_err)
    global_err = np.array(global_err)


    if verbose:
        # append 0 to the beginning of the group size
        group_size = [0] + group_size
        #power_offset = [[0] + x for x in power_offset]
        # reverse power offset
        #power_offset = np.array(power_offset)
        #power_offset = power_offset[:, ::-1]
        # reverse group offset
        group_offset = group_offset[::-1]

        plt.rcParams.update({'font.size': 10})

        # 3D plot
        fig = plt.figure(figsize=(15, 6))
        



            
       # print(group_size, power_offset)
        #print(np.array(power_offset).shape)
        
        # get global sim errors per group
        global_sim_err = np.array(global_sim_err)
        #print(global_sim_err.shape)
        # 14~ group sizes, 6~ group offsets 
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        #ax3 = fig.add_subplot(121)
        ax4 = fig.add_subplot(121)
        #perc names for ß1
        perc_names = np.array([' ~ 5%', ' ~ 7%', ' ~ 10%', ' ~ 12%', ' ~ 15%', ' ~ 19%'])[::-1]
        for i, offset in enumerate(group_offset):
            sim_means = global_sim_err[:,i,0]    
            sim_std = global_sim_err[:,i,1]
            #ax3.errorbar(group_size, sim_means, yerr= 1.96*sim_std , label='$ß_1$: ' + str(np.round(abs(offset[0]),2)), alpha=0.5, color = colors[i], marker='o')

            # plot the power curve for each offset and show the power as a second y axis
            l = len(group_offset)
            #print(l, i, l-1-i)
            #print(np.array(power_offset)[:,l-1-i].shape, np.array(group_size).shape)
            temp_power = np.array(power_offset)[:,l-1-i]

            # add a 0 to the beginning of the power list
            temp_power = np.insert(temp_power, 0, 0)
            
            ax4.plot(group_size, temp_power, alpha=0.5, label = '$ß_1$: ' + str(np.round(abs(offset[0]),2)) + perc_names[i], color = colors[i], marker='o')
 
            #ax3.plot(group_size, np.array(power_offset)[:,i], alpha=0.5, label = 'Power: ' + str(np.round(abs(offset[0]),2)))
        #ax3.set_xlabel('Group Size')
        #ax3.set_ylabel('Global Mean Error')
        ax4.set_ylabel('Power')
        ax4.set_xlabel('Group Size')
        ax4.legend()
        ax4.set_title('Power for Varying Group Ratio')
        #ax3.legend()
        plt.show()        
        

        #plt.savefig('ratio_sim.png', bbox_inches='tight', pad_inches=1)
        #plt.show()
    
    return global_err

# main
if __name__ == '__main__':
    np.random.seed(123)
    # group_size = [x for x in range(1, 501, 50)]
    # global_err = simulate_no_courses(no_courses = 50, # fix
    #                                 no_simulations = 100, 
    #                                 no_students = 5500, # fix
    #                                 group_size = group_size,   # influencial
    #                                 group_offset = [(0.1, -0.1), (0.15, -0.15), (0.2, -0.2), (0.25, -0.25),(0.3, -0.3), (0.4, -0.4), (0.5, -0.5),], # fix
    #                                 interpolation = 'linear',
    #                                 verbose=True)

    group_size = [x for x in range(50, 551, 50)]

    global_err = simulate_no_courses(no_courses = 50, # fix
                                    no_simulations = 1000, 
                                    no_students = 1000, # fix
                                    group_size = group_size,   # influencial
                                    group_offset = [(0.1, -0.1), (0.15, -0.15), (0.2, -0.2), (0.25, -0.25),(0.3, -0.3), (0.4, -0.4)], # fix
                                    interpolation = 'linear',
                                    verbose=True,
                                    offset = True)