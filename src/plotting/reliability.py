import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import os
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



def plot_reliability(degree, best_overall_model):

    # map mmod1_dim1 to 1pl_1dim, mmod2_dim1 to 2pl_1dim, mmod2_dim2 to 2pl_2dim, mmod2_dim3 to 2pl_3dim
    model_mapping = {'mmod1': ['1pl_1dim', 1], 'mmod2_dim1': ['2pl_1dim',1], 'mmod2_dim2': ['2pl_2dim',2], 'mmod2_dim3': ['2pl_3dim',3]}
    model = model_mapping[best_overall_model][0]
    dim = model_mapping[best_overall_model][1]

    data_folder = parent_dir + '/data/real/' + degree + '/'#
    tagged_data = np.array(pd.read_csv(data_folder + "binary_reduced/taggedRInput.csv", header=None))

    # add a column of 1s and 0s to the end of the data
    tagged_data = np.insert(tagged_data, tagged_data.shape[1], values=np.ones(tagged_data.shape[0]), axis=1)
    tagged_data = np.insert(tagged_data, tagged_data.shape[1], values=np.zeros(tagged_data.shape[0]), axis=1)

   
    
    #Randomly split the data into two halves
    reli_data = np.ones((tagged_data.shape[0], tagged_data.shape[1]*2))*-99999

    for stud in range(tagged_data.shape[1]):
        resp =  np.concatenate((np.where(tagged_data[:,stud]==0)[0], np.where(tagged_data[:,stud]==1)[0]))
        np.random.shuffle(resp)
        inds_1 = resp[:int(round(len(resp)/2))]
        inds_2 = resp[int(round(len(resp)/2)):] 
        reli_data[inds_1, stud] = tagged_data[inds_1, stud]
        reli_data[inds_2, stud+tagged_data.shape[1]] = tagged_data[inds_2, stud]


    


    reli_data = pd.DataFrame(reli_data)
    #data_test = pd.read_csv(folder + "binary/tagged_data_binary.csv", header=None)
    #Test wether reliability folder exists, if not create it
    if not os.path.exists(data_folder + 'reliability'):
        os.makedirs(data_folder + 'reliability')
    # print last student in reli_data
    reli_data.to_csv(data_folder + 'reliability/random_split.csv', header=None, index=False)


    #Split the data into two halves. First half vs second half of student responses.
    inds = []
    no_studs = 0
    for j in range(tagged_data.shape[1]):
        cur_dat = np.where(tagged_data[:,j]!=-99999)[0]
        if len(cur_dat)>10:
            no_studs+=1

    half_1 = np.ones((tagged_data.shape[0], no_studs))*-99999
    half_2 = np.ones((tagged_data.shape[0], no_studs))*-99999
    skip_count=0
    for j in range(tagged_data.shape[1]):
        cur_dat = np.where(tagged_data[:,j]!=-99999)[0]
        #Only use students with more than 18 responses otherwise timespan is too short.
        if len(cur_dat)>10:
            split = int(np.round(len(cur_dat)/2))
            inds_1 = cur_dat[:split]
            inds_2 = cur_dat[split:]
            half_1[:,j-skip_count][inds_1] = tagged_data[:,j][inds_1]
            half_2[:,j-skip_count][inds_2] = tagged_data[:,j][inds_2]
        else:
            skip_count+=1

    
    half_data = np.concatenate((half_1, half_2), axis=1)
    print('random reli shape', half_data.shape)
    np.savetxt(data_folder + 'reliability/time_split.csv', half_data, delimiter = ',')


    #Run the 1PL model on the two halves of the data through reli.R script
    analysis_dir = data_folder + '../../../src/analysis/'

    
    if '2pl' in model:
        item_type = '2PL'
    else:
        item_type = 'Rasch'

    if '2dim' in model:
        dof = 2
    elif '1dim' in model:
        dof = 1
    elif '3dim' in model:
        dof = 3
    else: 
        dof = 1
    ## Load reliability R script
    try:
        r = robjects.r
        r.source(analysis_dir + 'reli.R')
        r['compute_reli_estimates'](degree, item_type, dof)
        #Plot the results of random split
        fig_random = plt.figure(figsize=(8.5,5))
        reli_1 = np.array(pd.read_csv(data_folder + "reliability/random_1.csv")).flatten()
        reli_2 = np.array(pd.read_csv(data_folder + "reliability/random_2.csv")).flatten()
        r2 = scipy.stats.pearsonr(reli_1,reli_2)
        print(np.sqrt(np.mean((reli_1-reli_2)**2)))
        plt.scatter(reli_1, reli_2, alpha=0.5, color='black')
        plt.xlim((-4.2,4.2))
        plt.ylim((-4.2,4.2))
        plt.xlabel('Abilities')
        plt.ylabel('Abilities')

        plt.plot([-4.2,4.2],[-4.2,4.2], color='black', linestyle='--')
        plt.title("pearson r = " + str("%.3f" % r2[0]) + ", p < " + str("%.3f" % 0.001))
        plt.savefig(data_folder + 'plots/random_reli.pdf', format="pdf")
        plt.show()

        fig_time = plt.figure(figsize=(8.5,5))
        reli_1 = np.array(pd.read_csv(data_folder + "reliability/time_1.csv")).flatten()
        reli_2 = np.array(pd.read_csv(data_folder + "reliability/time_2.csv")).flatten()
        r2 = scipy.stats.pearsonr(reli_1,reli_2)
        print(np.sqrt(np.mean((reli_1-reli_2)**2)))
        plt.scatter(reli_1, reli_2, alpha=0.5, color='black')
        
        plt.plot([-4.2,4.2],[-4.2,4.2], color='black', linestyle='--')
        plt.xlim((-4.2,4.2))
        plt.ylim((-4.2,4.2))
        plt.xlabel('Abilities')
        plt.ylabel('Abilities')
        plt.title("pearson r = " + str("%.3f" % r2[0]) + ", p < " + str("%.3f" % 0.001))
        plt.savefig(data_folder + 'plots/time_reli.pdf', format="pdf")
        plt.show()
        return fig_random, fig_time
    except Exception as e:
        print(e)
        print("Error in reliability plotting")
        return None, None

if __name__ == '__main__':
    reliability(degree = 'CompSci')

