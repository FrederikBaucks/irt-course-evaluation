import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


def plot_loadings(degree, best_overall_model):
    model_mapping = {'mmod1': ['mmod1', 1], 'mmod2_dim1': ['mmod2_dim1',1], 'mmod2_dim2': ['mmmod2_dim2',2], 'mmod2_dim3': ['mmmod2_dim3',3]}
    model = model_mapping[best_overall_model][0]
    dim = model_mapping[best_overall_model][1]
    # Get current directory
    current_dir = os.getcwd()
    # Get parent directory
    parent_dir = os.path.dirname(current_dir)
    folder = parent_dir + '/data/real/' + degree + '/irt_results/models/model_loadings/'
    
    # Get all files in folder
    loading_files = os.listdir(folder)
    fig = plt.figure(figsize=(5, 12))
    for loadings in loading_files:
        # Get the loadings
        file = folder + loadings
        with open(file) as f:
            lines = f.readlines()
        # Search lines for Factors
        if 'dim2' in loadings or 'dim3' in loadings:
            rows2skip = 5
        else:
            rows2skip = 0
            continue

        factor_line = lines[rows2skip]

        # prop of var last line of lines
        prop_var = lines[-1]
        factors = [x+factor_line[ind+1] for ind, x in enumerate(factor_line) if x == 'F']
        
        # Transform to dataframe
        df_loadings = pd.read_csv(file, sep='\s+', skiprows=rows2skip, skipfooter=9, engine='python')

        
        # Plot loadings with dimensions according to the number of factors
    
        if 'dim2' in loadings:
            ax1 = fig.add_subplot(2, 1, 1)
            ax1.scatter(df_loadings['F1'], df_loadings['F2'])
            ax1.set_xlabel('F1')
            ax1.set_ylabel('F2')
            ax1.set_title('Loadings for ' + degree + ' with 2 factors: ' + prop_var)
        else:
            ax2 = fig.add_subplot(2, 1, 2, projection='3d')
            ax2.scatter(df_loadings['F1'], df_loadings['F2'], df_loadings['F3'])
            ax2.set_xlabel('F1')
            ax2.set_ylabel('F2')
            ax2.set_zlabel('F3')
            ax2.set_title('Loadings for ' + degree + ' with 3 factors: ' + prop_var)
    
    plt.savefig(parent_dir + '/data/real/' + degree + '/plots/loadings.pdf', format='pdf', pad_inches=2, bbox_inches='tight')
    plt.show()  
    return None


if __name__ == "__main__":
    degree = 'CompSci'
    plot_loadings(degree)