import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# 3d plot with group ratio and group offset and error
from mpl_toolkits.mplot3d import Axes3D

from scipy.interpolate import griddata
def simulate_size(no_courses, no_students, no_simulations, group_size = [x/500 for x in range(1, 70, 10)], group_offset = [(x, y) for x in np.arange(-2, 0, 1) for y in np.arange(1, 3, 1)], interpolation='linear', verbose=False):

    # Generate IRT parameters
    theta = np.random.normal(0, 1, no_students)
    d = np.random.normal(0, 1, no_courses)
    d = d.reshape((no_courses, 1))

    # Generate student responses
    p = 1 / (1 + np.exp(-(theta - d)))
    y = np.random.binomial(1, p)

    # DCF course 
    dcf = 0

    
    # Group offsets
    

    #sim_err = []
    global_sim_err = []
    global_err = []
    global_intercepts = []
    for g_size in group_size:
        errors = []
        intercepts = []
        sim_err = []
        
        for offset in group_offset:
            current_sim_err = []
            for sim in range(no_simulations):
                # split students into two groups based on group ratio

                
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
                offset_data = data['theta'] - data['difficulty'] 
                model = glm(reg_form, data=data, family=sm.families.Binomial(), offset=offset_data)
                intercept = model.fit().params[0]
                group = model.fit().params[1]
                p_pred_1 = 1 / (1 + np.exp(-(theta[group1] - d[dcf][0] + intercept - group))) 
                p_pred_2 = 1 / (1 + np.exp(-(theta[group2] - d[dcf][0] + intercept + group)))   
                err = np.mean(np.mean(np.abs(p_pred_1 - p_1)) + np.mean(np.abs(p_pred_2 - p_2)))
                

                
                current_sim_err.append(err)
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
        global_err.append(errors)
        global_intercepts.append(intercepts)
        global_sim_err.append(sim_err)
    global_err = np.array(global_err)


    if verbose:
        plt.rcParams.update({'font.size': 10})

        # 3D plot
        fig = plt.figure(figsize=(15, 6))
        ax1 = fig.add_subplot(141, projection='3d')

        X, Y = np.meshgrid(np.array(group_offset)[:,0], np.array(group_size))
        Z = np.array(global_err)
        print(X.shape, Y.shape, Z.shape)
        ax1.scatter(X.flatten(), Y.flatten(), Z.flatten(), c=Z.flatten(), cmap='Spectral', marker='o', alpha=0.3)
        #ax.plot_surface(X, Y, Z, cmap='viridis')

        ax1.set_xlabel('Group Offset')
        ax1.set_ylabel('Group Ratio')
        ax1.set_zlabel('Global Mean Error')
        ax1.view_init(elev=30, azim=35)
        #plt.savefig('sim_ratio_-1.png', dpi=300, bbox_inches='tight')


        x = np.array([group_size for x in range(0, len(group_offset))]).flatten()
        y = np.array([group_offset for x in range(0, len(group_size))])[:,:,0].flatten()
        z = global_err.flatten()

        X, Y, Z = [], [], []
        for x, x_val in enumerate(group_offset):
            for y, y_val in enumerate(group_size):
                    
                    X.append(x_val)
                    Y.append(y_val)
                    Z.append(global_err[y,x])
        X, Y, Z = np.array(X)[:,0].flatten(), np.array(Y).flatten(), np.array(Z).flatten()

        unique_XY = []
        average_Z = []
        for i in range(len(X)):
            point = (X[i], Y[i])
            # find unique x, y pairs in X, Y:
            if point not in unique_XY:
                unique_XY.append(point)

                # Find indices where this x,y pair occurs
                indices = [j for j, p in enumerate(zip(X, Y)) if p == point]

                # Calculate the mean of the corresponding z values
                mean_z = np.mean([Z[k] for k in indices])
                average_Z.append(mean_z)

        # Convert the unique points and their average z values to numpy arrays
        new_X, new_Y = np.array(unique_XY).T
        new_Z = np.array(average_Z)

        # Interpolate the data to create a smooth surface
        # We need to flatten the arrays to pass to 'griddata'
        xi = np.linspace(new_X.min(), new_X.max(), 1000)
        yi = np.linspace(new_Y.min(), new_Y.max(), 1000)
        xi, yi = np.meshgrid(xi, yi)
        zi = griddata((new_X, new_Y), new_Z, (xi, yi), method=interpolation)

        # 3D plot
        #fig = plt.figure(figsize=(13, 5))
        #ax = fig.add_subplot(121, projection='3d')

        # Plot the surface
        ax1.plot_surface(xi, yi, zi, cmap='Spectral', alpha=0.8)

        ax1.set_xlabel('Group Offset')
        ax1.set_ylabel('Group Ratio')
        ax1.set_zlabel('Global Mean Error')
        ax1.set_zlim(0, 1)
        ax1.view_init(elev=30, azim=25)


    
        ax2 = fig.add_subplot(142) 



            

        
        # get global sim errors per group
        global_sim_err = np.array(global_sim_err)
        print(global_sim_err.shape)
        # 14~ group sizes, 6~ group offsets 
        
        ax3 = fig.add_subplot(143)
        for i, offset in enumerate(group_offset):
            sim_means = global_sim_err[:,i,0]    
            sim_std = global_sim_err[:,i,1]
            ax3.errorbar(group_size, sim_means, yerr= 1.96*sim_std , label='Offset: ' + str(np.round(abs(offset[0]),2)), alpha=0.5)

        ax3.set_xlabel('Group Size')
        ax3.set_ylabel('Global Mean Error')
        ax3.legend()

        

        #plt.savefig('ratio_sim.png', bbox_inches='tight', pad_inches=1)
        #plt.show()
    
    return global_err


def simulate_ratio(no_courses, no_students, group_ratio = [x/500 for x in range(1, 70, 10)], group_offset = [(x, y) for x in np.arange(-2, 0, 1) for y in np.arange(1, 3, 1)], interpolation='linear', verbose=False):

    # Generate IRT parameters
    theta = np.random.normal(0, 1, no_students)
    d = np.random.normal(0, 1, no_courses)
    d = d.reshape((no_courses, 1))

    # Generate student responses
    p = 1 / (1 + np.exp(-(theta - d)))
    y = np.random.binomial(1, p)

    # DCF course 
    dcf = 0

    
    group_size = [10, 20, 30, 40, 80]
    # Group offsets
    

   
    global_err = []
    global_intercepts = []
    for g_size in group_ratio:
        errors = []
        intercepts = []
        for offset in group_offset:
            # split students into two groups based on group ratio
            group1 = np.random.choice(np.arange(0, no_students), int(no_students*g_size), replace=False)#no_students*group_ratio)
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
            offset_data = data['theta'] - data['difficulty'] 
            model = glm(reg_form, data=data, family=sm.families.Binomial(), offset=offset_data)
            intercept = model.fit().params[0]
            group = model.fit().params[1]
            p_pred_1 = 1 / (1 + np.exp(-(theta[group1] - d[dcf][0] + intercept - group))) 
            p_pred_2 = 1 / (1 + np.exp(-(theta[group2] - d[dcf][0] + intercept + group)))   
            err = np.mean(np.mean(np.abs(p_pred_1 - p_1)) + np.mean(np.abs(p_pred_2 - p_2)))
            

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

        global_err.append(errors)
        global_intercepts.append(intercepts)
    global_err = np.array(global_err)


    if verbose:
        plt.rcParams.update({'font.size': 10})

        # 3D plot
        fig = plt.figure(figsize=(15, 6))
        ax1 = fig.add_subplot(121, projection='3d')

        X, Y = np.meshgrid(np.array(group_offset)[:,0], np.array(group_ratio))
        Z = np.array(global_err)
        print(X.shape, Y.shape, Z.shape)
        ax1.scatter(X.flatten(), Y.flatten(), Z.flatten(), c=Z.flatten(), cmap='Spectral', marker='o', alpha=0.3)
        #ax.plot_surface(X, Y, Z, cmap='viridis')

        ax1.set_xlabel('Group Offset')
        ax1.set_ylabel('Group Ratio')
        ax1.set_zlabel('Global Mean Error')
        ax1.view_init(elev=30, azim=35)
        #plt.savefig('sim_ratio_-1.png', dpi=300, bbox_inches='tight')



        #fig = plt.figure(figsize=(8, 20))
        ax2 = fig.add_subplot(122, projection='3d')

        X, Y = np.meshgrid(np.array(group_offset)[:,1], 1-np.array(group_ratio))
        Z = np.array(global_err)
        ax2.scatter(X.flatten(), Y.flatten(), Z.flatten(), c=Z.flatten(), cmap='Spectral', marker='o', alpha=0.3)
        #ax.plot_surface(X, Y, Z, cmap='viridis')

        ax2.set_xlabel('Group Offset')
        ax2.set_ylabel('Group Ratio')
        ax2.set_zlabel('Global Mean Error')
        ax2.view_init(elev=30, azim=-150)
        # make the text fit in the plot
        #plt.savefig('sim_ratio_1.png', dpi=300, bbox_inches='tight')
        #plt.show()


        x = np.array([group_ratio for x in range(0, len(group_offset))]).flatten()
        y = np.array([group_offset for x in range(0, len(group_ratio))])[:,:,0].flatten()
        z = global_err.flatten()

        X, Y, Z = [], [], []
        for x, x_val in enumerate(group_offset):
            for y, y_val in enumerate(group_ratio):
                    
                    X.append(x_val)
                    Y.append(y_val)
                    Z.append(global_err[y,x])
        X, Y, Z = np.array(X)[:,0].flatten(), np.array(Y).flatten(), np.array(Z).flatten()

        unique_XY = []
        average_Z = []
        for i in range(len(X)):
            point = (X[i], Y[i])
            # find unique x, y pairs in X, Y:
            if point not in unique_XY:
                unique_XY.append(point)

                # Find indices where this x,y pair occurs
                indices = [j for j, p in enumerate(zip(X, Y)) if p == point]

                # Calculate the mean of the corresponding z values
                mean_z = np.mean([Z[k] for k in indices])
                average_Z.append(mean_z)

        # Convert the unique points and their average z values to numpy arrays
        new_X, new_Y = np.array(unique_XY).T
        new_Z = np.array(average_Z)

        # Interpolate the data to create a smooth surface
        # We need to flatten the arrays to pass to 'griddata'
        xi = np.linspace(new_X.min(), new_X.max(), 1000)
        yi = np.linspace(new_Y.min(), new_Y.max(), 1000)
        xi, yi = np.meshgrid(xi, yi)
        zi = griddata((new_X, new_Y), new_Z, (xi, yi), method=interpolation)

        # 3D plot
        #fig = plt.figure(figsize=(13, 5))
        #ax = fig.add_subplot(121, projection='3d')

        # Plot the surface
        ax1.plot_surface(xi, yi, zi, cmap='Spectral', alpha=0.8)

        ax1.set_xlabel('Group Offset')
        ax1.set_ylabel('Group Ratio')
        ax1.set_zlabel('Global Mean Error')
        ax1.set_zlim(0, 1)
        ax1.view_init(elev=30, azim=25)


        print(global_err.shape, len(group_ratio), len(group_offset))
        x = np.array([group_ratio for x in range(0, len(group_offset))]).flatten()
        y = np.array([group_offset for x in range(0, len(group_ratio))])[:,:,1].flatten()
        z = global_err.flatten()

        X, Y, Z = [], [], []
        for x, x_val in enumerate(group_offset):
            for y, y_val in enumerate(group_ratio):
                    
                    X.append(x_val)
                    Y.append(1-y_val)
                    Z.append(global_err[y,x])
        




        X, Y, Z = np.array(X)[:,1].flatten(), np.array(Y).flatten(), np.array(Z).flatten()
        print('old', X.shape, Y.shape, Z.shape)
        unique_XY = []
        average_Z = []
        for i in range(len(X)):
            point = (X[i], Y[i])
            # find unique x, y pairs in X, Y:
            if point not in unique_XY:
                unique_XY.append(point)

                # Find indices where this x,y pair occurs
                indices = [j for j, p in enumerate(zip(X, Y)) if p == point]

                # Calculate the mean of the corresponding z values
                mean_z = np.mean([Z[k] for k in indices])
                average_Z.append(mean_z)

        # Convert the unique points and their average z values to numpy arrays
        new_X, new_Y = np.array(unique_XY).T
        new_Z = np.array(average_Z)
        print('new', new_X.shape, new_Y.shape, new_Z.shape)
            # find the mean of the z values
            # append to new X, Y, Z

            
        # Interpolate the data to create a smooth surface
        # We need to flatten the arrays to pass to 'griddata'
        xi = np.linspace(new_X.min(), new_X.max(), 1000)
        yi = np.linspace(new_Y.min(), new_Y.max(), 1000)
        xi, yi = np.meshgrid(xi, yi)
        zi = griddata((new_X, new_Y), new_Z, (xi, yi), method=interpolation)

        # 3D plot
        #fig = plt.figure(figsize=(8, 10))
        #ax = fig.add_subplot(122, projection='3d')

        # Plot the surface
        ax2.plot_surface(xi, yi, zi, cmap='Spectral', alpha=0.8)
        ax2.set_zlim(0, 1)
        ax2.set_xlabel('Group Offset')
        ax2.set_ylabel('Group Ratio')
        ax2.set_zlabel('Global Mean Error')
        ax2.view_init(elev=30, azim=-160)
        plt.subplots_adjust(left=.1, right=0.8, bottom=0.1, top=0.9)
        # Show the plot
        
        plt.savefig('ratio_sim.png', bbox_inches='tight', pad_inches=1)
        plt.show()
    return global_err
