# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ## Import Main Librairies

# +
# Basic Python modules
import random
import csv
import pickle
import os

# Scientific modules
import numpy as np
import scipy as scp
import matplotlib.pyplot as plt
from scipy.io import loadmat
import networkx as nx

# For better looking graphs
import seaborn as sns
sns.set()

# Just a custom color palette that I use
# blue, darkblue, red, orange, green, palegreen, yellow, brokenwhite, brokengrey
colors = ["#AF3127","#6182B5","#112A3C","#D99C37","#90A954","#C5B868","#FAC764","#DAC0A6","#C4C2D1"]
sns.set_palette(sns.color_palette(colors))
# -

# ## Configure paths for data once for all

# +
#generate_images = False

#pickle_data_folder = 'new_new_pickle_data_cross/'

# Folder in which data is placed
data_folder = '../fake_data'

# Suffix at the end of each .mat file; it is specified here in order to select 
# relevant files, as well as to make file name lighter during loading for 
# further operations (such as printing subjects names), since it does not carry 
# additional information.
suffix = '_fiber_number.mat'

# For instance here, with those setting, every ../fake_data/*_fiber_number.mat 
# will be loaded

# Keys used to split data between patients and controls. Subject whose filename 
# contains one of the control_keys will be affected to the control cohort, and 
# similarly for patients.
control_keys = ['060', 'dep', 'dpr', 'S', 'TI']
patient_keys = ['lgp']

# By default, the code expects a "table.csv" present in data_folder, containing 
# information about the patients, such as their age, the duration of the 
# disease, etc.
csv_path = data_folder + "/table.csv"


# -

# ## Load data

# +
def get_matrix_file_list(data_folder, suffix):
    """ 
        Return the list of files in the folder data_folder ending by suffix 
    """
    file_list = [f for f in os.listdir(data_folder) if f.endswith(suffix)]
    return list(map(lambda x: data_folder + '/' + x, file_list))

def load_matrix_file(file):
    """ 
        Return the matrix loaded from a Matlab data file. Note that the 
        'Measure' key is hardcoded, so you might need to adjust it to your own 
        data.
    """
    return loadmat(file)['Measure']

# Create a dictionnary of all the matrices, where each matrix gets associated to 
# the filename of the corresponding .mat file minus the suffix.
connectivity_matrices = {}

for f in get_matrix_file_list(data_folder, suffix):
    connectivity_matrices[f.replace(suffix,'').replace(data_folder + '/','')] = load_matrix_file(f)
    
# Create a dictionnary of metadata for each patient, obtained from the file 
# at csv_path (by default 'data_foler/table.csv')
patient_info_dict = {}

with open(csv_path, 'r') as csv_file:
    metadata = csv.DictReader(csv_file)
    # Each patient is associated to a dictionnary containing all its information
    for row in metadata:
        metadata_dict = {key:row[key] for key in metadata.fieldnames if key != 'Subject'}
        patient_info_dict[row['Subject']] = metadata_dict
        
print("Succesfully loaded {} matrices from {}.".format(len(connectivity_matrices), data_folder))
print("Metadata has been found in {} for {} subjects.".format(csv_path,len(patient_info_dict)))
# -

# ## Split Data into Cohorts

# +
# list of controls and patients names
controls = []
patients = []

# The following can be used to limit the number of either controls or patients 
# considered for the study if they are set to some non infinite number.
controls_count = np.inf
patients_count = np.inf

current_control = 0
current_patient = 0

for key in [*connectivity_matrices]:
    # Use patients_keys and control_keys list to classify subject into cohorts
    if any(list(map(lambda x: x in key, patient_keys))) and current_patient < patients_count:
        patients.append(key)
        current_patient += 1
    elif any(list(map(lambda x: x in key, control_keys))) and current_control < controls_count:
        controls.append(key)
        current_control += 1
    else:
        print("Patient {} cannot be classified either as control or patient.".format(key))

controls_count = current_control
patients_count = current_patient

subject_count = len(patients) + len(controls)

print("Classified {} controls and {} patients (total {} subjects)".format(controls_count, patients_count, subject_count))


# -

# ## Basic network manipulation functions

# +
def plot_connectivity_matrix(*matrix, save=None):
    """
        Plot the connectivity matrix as a colormap
        It uses a logarithmic scale for the colors and it is possible to plot 
        multiple matrices at once.
    """
    
    max_rows = 2
    line = len(matrix) / 3 + 1
    
    for i, mat in enumerate(matrix):
        plt.subplot(line, max_rows, i+1)
        plt.imshow(np.log(mat+1),cmap='viridis')
    
    plt.tight_layout()
    if save is None:
        plt.show()
    else:
        plt.savefig(save,bbox_inches='tight')
        
def get_network(matrix, threshold = 0):
    """ 
        Return the network (as a networkx data structure) defined by matrix.
        It is possible to specify a threshold that will disregard all the 
        edges below this threshold when creating the network
    """
    G = nx.Graph()
    N = matrix.shape[0]
    G.add_nodes_from(list(range(N)))
    G.add_weighted_edges_from([(i,j,1.0*matrix[i][j]) for i in range(0,N) for j in range(0,i) \
                                                                   if matrix[i][j] >= threshold])
    return G

def filter_weights(matrix, ratio):
    """
        Only keep a fraction of the weights of the matrix, fraction specified 
        via the threshold parameter
    """
    n = matrix.shape[0]
    filtered_matrix = np.zeros_like(matrix)
    total_weight = np.sum(matrix)
    weights_id = sorted([(matrix[i,j],i,j) for i in range(n-1) for j in range(i+1,n)], reverse=True)
    
    filtered_weight = 0
    for (w,i,j) in weights_id:
        filtered_weight += 2*w
        
        if filtered_weight > ratio*total_weight:
            break
        
        filtered_matrix[i,j] = w
        filtered_matrix[j,i] = w
    
    return filtered_matrix


# -

# ## Laplacian manipulation

# +
def get_laplacian_matrix(matrix, threshold = 0):
    """ 
        Return the laplacian matrix of the network defined by matrix.
        It is possible to specify a threshold that will disregard all the 
        edges below this threshold when creating the network
    """
    
    L = np.zeros_like(matrix)
    
    assert(L.shape[0] == L.shape[1]), "The network should be encoded by a square matrix"
    
    N = L.shape[0]
    
    # Small one liner trick to get rid of the diagonal and apply the threshold.
    A = np.multiply(np.where(matrix >= threshold, matrix , np.zeros_like(matrix)), np.ones((N,N)) - np.eye(N))
    
    # Use the definition of L as L = D - A 
    L = np.eye(N) * np.array([np.sum(A[i,:]) for i in range(N)]) - A
    
    return L

def get_stabilized_laplacian_matrix(matrix, threshold = 0, ratio=1.0, alpha = 1e-2):
    """
        Prefered function to get the laplacian matrix of a network encoded by a 
        matrix. Can apply a ratio filtering of the weights of the matrix (i.e. 
        only keep ratio of the total weight) or / and apply a threshold. The 
        alpha parameter corrects a numerical issue.
    """
    L = get_laplacian_matrix(filter_weights(matrix, ratio = ratio), threshold)
    
    n = L.shape[0]
    
    # Add a small epsilon to the diagonal of the Laplacian matrix. It should 
    # make the computation more stable. Idea from https://github.com/Hermina/GOT
    
    return L + alpha * np.eye(n)


# -

# Small tests to assess consistency of the different functions.
assert np.allclose(nx.linalg.laplacian_matrix(get_network(connectivity_matrices[controls[0]])).toarray(), get_laplacian_matrix(connectivity_matrices[controls[0]])), "Inconsistency in the Laplacian functions"
assert np.allclose(nx.linalg.laplacian_matrix(get_network(connectivity_matrices[patients[0]], threshold=10)).toarray(), get_stabilized_laplacian_matrix(connectivity_matrices[patients[0]], alpha=0, threshold=10)), "Inconsistency in the Laplacian functions (test 2)"

# ## Wasserstein distance functions

# +
# This is a dictionary used to speed up computations, by avoiding multiple 
# computations of the same objects.
precomp_dict_net = {}
precomp_dict_cov = {}

def wasserstein_networks(A, B, threshold = 0, threshold2 = None, ratio = 1.0):
    """ 
            Returns the Wasserstein distance (in the sense of the GOT paper) 
            between the two networks encoded by A and B. It is possible to 
            apply a threshold to the network obtained via both matrices (and 
            even a different one per matrices)
    """
    
    if threshold2 is None:
        threshold2 = threshold
    
    assert(A.shape == B.shape), "compared networks should have the \
                                                        same number of nodes"
    
    L1dag = None
    L1dag_sqrt = None
    L2dag = None
    L2dag_sqrt = None
    
    if (A.tobytes(), threshold, ratio) not in precomp_dict_net:
        L1 = get_stabilized_laplacian_matrix(A, threshold = threshold, ratio = ratio)
        L1dag = np.real(scp.linalg.pinv(L1))
        L1dag_sqrt = np.real(scp.linalg.sqrtm(L1dag))
        
        precomp_dict_net[(A.tobytes(), threshold,ratio)] = (L1dag, L1dag_sqrt)
    else:
        L1dag, L1dag_sqrt = precomp_dict_net[(A.tobytes(), threshold, ratio)]
    
    if (B.tobytes(), threshold2, ratio) not in precomp_dict_net:
        L2 = get_stabilized_laplacian_matrix(B, threshold = threshold2, ratio = ratio)
        L2dag = np.real(scp.linalg.pinv(L2))
        L2dag_sqrt = np.real(scp.linalg.sqrtm(L2dag))
        
        precomp_dict_net[(B.tobytes(), threshold2, ratio)] = (L2dag, L2dag_sqrt)
    
    else :
        L2dag, L2dag_sqrt = precomp_dict_net[(B.tobytes(), threshold2, ratio)]

    W = np.trace(L1dag) + np.trace(L2dag) - 2*np.trace(np.real(scp.linalg.sqrtm(L1dag_sqrt @ L2dag @ L1dag_sqrt)))
    
    return W


def wasserstein_network_covariance(A, cov, threshold = 0, ratio = 1.0):
    """ 
            Return the Wasserstein distance (in the sense of the GOT paper) 
            between the the network encoded by a matrix A, and the normal 
            distribution whose covariance matrix is given by cov. 
            It is possible to apply a threshold to the network obtained via 
            matrix A.
    """
    
    assert(A.shape == cov.shape), "compared networks should have the \
                                                        same number of nodes"
    
    N = A.shape[0]

    L1dag = None
    L1dag_sqrt = None
    L2dag = None
    L2dag_sqrt = None
    
    if (A.tobytes(), threshold, ratio) not in precomp_dict_net:
        L1 = get_stabilized_laplacian_matrix(A, threshold = threshold, ratio = ratio)
        L1dag = np.real(scp.linalg.pinv(L1))
        L1dag_sqrt = np.real(scp.linalg.sqrtm(L1dag))
        
        precomp_dict_net[(A.tobytes(), threshold ,ratio)] = (L1dag, L1dag_sqrt)
    else:
        L1dag, L1dag_sqrt = precomp_dict_net[(A.tobytes(), threshold ,ratio)]
    
    if cov.tobytes() not in precomp_dict_cov:
        L2dag = cov
        L2dag_sqrt = np.real(scp.linalg.sqrtm(L2dag))
        
        precomp_dict_cov[cov.tobytes()] = L2dag_sqrt
    
    else:
        L2dag = cov
        L2dag_sqrt = precomp_dict_cov[cov.tobytes()]
    
    W = np.trace(L1dag) + np.trace(L2dag) - 2*np.trace(np.real(scp.linalg.sqrtm(L1dag_sqrt @ L2dag @ L1dag_sqrt)))
    return W

def clip_remove(array, threshold):
    """
        Small helper function to clip an array by removing values above a 
        certain threshold
    """
    r = array[array < threshold]
    print("Kept {} out of {} ({} %)".format(r.shape[0], array.shape[0], r.shape[0]/array.shape[0] * 100))
    return r

def barycenter_iteration(cov_list, iteration_count = 10):
    """
        Return the covariance matrix of the barycenter of the distributions 
        whose covariances are given as cov_list ; iteration_count controls the
        number of iteration made by the iterative algorithm.
    """
    covariance_count = len(cov_list)

    size = cov_list[0].shape[0]
    curr_cov = np.eye(size)
    
    for iteration in range(iteration_count):
        curr_cov_sqrt = np.real(scp.linalg.sqrtm(curr_cov))
        curr_cov_pinv_sqrt = scp.linalg.pinv(curr_cov_sqrt)
        big_sum = np.zeros_like(curr_cov)
        for sigma in cov_list:
            big_sum += 1/covariance_count * np.real(scp.linalg.sqrtm(curr_cov_sqrt @ sigma @ curr_cov_sqrt))
        big_sum = big_sum @ big_sum
        curr_cov = curr_cov_pinv_sqrt @ big_sum @ curr_cov_pinv_sqrt
    
    return curr_cov

# Sum functions to print a barycenter as an approximate connectivity matrix
def cov_to_laplacian(cov, alpha = alpha):
    n = cov.shape[0]
    return scp.linalg.pinv(cov) - alpha * np.eye(n) #- np.ones_like(cov)/n

def laplacian_to_connectivity_matrix(laplacian):
    n = laplacian.shape[0]
    return np.multiply(laplacian, -(np.ones_like(laplacian) - 2*np.eye(n)))


# -

# ## Perform a cross validation

def subsample_controls(controls, sample_size, ratio = 1.0):
    # number of controls used to compute the barycenter
    #controls_barycenter_count = 25
    controls_count = len(controls)

    random_controls_sample = random.sample(range(0, controls_count),sample_size)
    controls_barycenter = np.array(controls)[random_controls_sample]

    pseudo_inverses_first_hemispher = [scp.linalg.pinv(get_stabilized_laplacian_matrix(first_hemispher_dict[subject],ratio=ratio)) for subject in controls_barycenter]
    pseudo_inverses_second_hemispher = [scp.linalg.pinv(get_stabilized_laplacian_matrix(second_hemispher_dict[subject],ratio=ratio)) for subject in controls_barycenter]
    
    return (controls_barycenter, pseudo_inverses_first_hemispher, pseudo_inverses_second_hemispher)


def do_stuff():
    random_fold_count = 0

    control_sample_size = 10

    pickle_folder = "new_new_pickle_data_folding/"

    for ratio in [0.55,0.6,0.7,0.8,0.9,1.0]:#[0.1,0.15,0.2,0.25,0.3,0.35,0.40,0.45,0.5]:#,0.55,0.6,0.65,0.7,0.75,0.8]:
        true_folds = 0
        print("Folding with ratio {:.2f}".format(ratio))

        pickle_file_path = pickle_folder + "dist_and_count_r{:.2f}_k{}.pickle".format(ratio, control_sample_size)
        print("Will save or update {}".format(pickle_file_path))

        distance_dict = {s : 0.0 for s in patients + controls}
        distance_fold_count = {s:0 for s in patients + controls}

        for fold in range(random_fold_count):
            if fold % 10 == 0: print("Fold {}".format(fold))

            controls_sample, pinvs_first_hemispher, pinvs_second_hemispher = subsample_controls(controls, control_sample_size, ratio = ratio)

            barycenter_cov_first_hemispher = barycenter_iteration(pinvs_first_hemispher)
            barycenter_cov_second_hemispher = barycenter_iteration(pinvs_second_hemispher)

            subjects = patients + [c for c in controls if c not in controls_sample]
            distance_dict_first_hemisphere = {subject:wasserstein_network_covariance(first_hemispher_dict[subject], barycenter_cov_first_hemispher, ratio = ratio) for subject in subjects}
            distance_dict_second_hemisphere = {subject:wasserstein_network_covariance(second_hemispher_dict[subject], barycenter_cov_first_hemispher, ratio = ratio) for subject in subjects}

            for subject, d in distance_dict_first_hemisphere.items():
                distance_dict[subject] += (d + distance_dict_second_hemisphere[subject])
                distance_fold_count[subject] += 1

            true_folds += 1

        if os.path.exists(pickle_file_path):
            with open(pickle_file_path, 'rb') as filehandler:
                loaded_dist_dict, loaded_dist_count, loaded_fold_count = pickle.load(filehandler)

                for s in patients + controls:
                    distance_dict[s] += loaded_dist_dict[s]
                    distance_fold_count[s] += loaded_dist_count[s]

                print("{} ({} folds) will be updated with {} more folds".format(pickle_file_path, loaded_fold_count, random_fold_count))
                true_folds += loaded_fold_count

        with open(pickle_file_path, 'wb') as filehandler:
            print("Saving {}".format(pickle_file_path))
            pickle.dump((distance_dict, distance_fold_count, true_folds), filehandler)

        dist_list = []
        for s in patients:
            dist_list.append(distance_dict[s] / distance_fold_count[s])

        for c in controls:
            dist_list.append(distance_dict[c] / distance_fold_count[c])  

        print(scp.stats.ttest_ind(dist_list[:patients_count], dist_list[patients_count:], equal_var=True))
        
        threshold = 2*np.mean(dist_list)
        print("Plot thresholded at {}".format(threshold))
        plt.hist(np.clip(dist_list[:patients_count],None, threshold), stacked=True, density=True, range=(0,threshold),bins=30, alpha=0.3, label='Patients')
        plt.hist(np.clip(dist_list[patients_count:],None, threshold), stacked=True, density=True, range=(0,threshold), bins=30, alpha = 0.3, label='Controls')
        sns.kdeplot(np.clip(dist_list[:patients_count],None,threshold), shade=False, color='C0', alpha=0.3);
        sns.kdeplot(np.clip(dist_list[patients_count:],None,threshold), shade=False, color='C1', alpha=0.3);
        plt.xlabel("Distance to barycenter ({:.2f})".format(ratio))
        plt.legend()
        #plt.show()
        
        import tikzplotlib
        tikzplotlib.save("figures_generation/barycenter_fold_mean_alpha_{:.2f}.tex".format(ratio), wrap=False)
        plt.show()


do_stuff()


def get_results(ratio, control_sample_size):
    pickle_folder = 'new_new_pickle_data_folding/'
    pickle_file_path = pickle_folder + "dist_and_count_r{:.2f}_k{}.pickle".format(ratio, control_sample_size)
    
    subjects = patients + controls
    distance_dict = {s: 0.0 for s in subjects}
    distance_fold_count = {s: 0 for s in subjects}
    
    if os.path.exists(pickle_file_path):
        with open(pickle_file_path, 'rb') as filehandler:
            loaded_dist_dict, loaded_dist_count, loaded_fold_count = pickle.load(filehandler)

            for s in patients + controls:
                distance_dict[s] += loaded_dist_dict[s]
                distance_fold_count[s] += loaded_dist_count[s]

    dist_list = []
    for s in patients:
        dist_list.append(distance_dict[s] / distance_fold_count[s])

    for c in controls:
        dist_list.append(distance_dict[c] / distance_fold_count[c])  

    print(scp.stats.ttest_ind(dist_list[:patients_count], dist_list[patients_count:], equal_var=True))

    threshold = 2*np.mean(dist_list)
    print("Plot thresholded at {}".format(threshold))
    plt.hist(np.clip(dist_list[:patients_count],None, threshold), stacked=True, density=True, range=(0,threshold),bins=30, alpha=0.3, label='Patients')
    plt.hist(np.clip(dist_list[patients_count:],None, threshold), stacked=True, density=True, range=(0,threshold), bins=30, alpha = 0.3, label='Controls')
    sns.kdeplot(np.clip(dist_list[:patients_count],None,threshold), shade=False, color='C0', alpha=0.3);
    sns.kdeplot(np.clip(dist_list[patients_count:],None,threshold), shade=False, color='C1', alpha=0.3);
    plt.xlabel("Distance to barycenter ({:.2f})".format(ratio))
    plt.legend()
    plt.savefig("distance_to_barycenter.svg")
    #plt.show()


get_results(1, 10)


# # Same thing, but with full brain

# +
def cov_to_laplacian(cov, alpha = 1e-2):
    n = cov.shape[0]
    return scp.linalg.pinv(cov) - alpha * np.eye(n) #- np.ones_like(cov)/n

def laplacian_to_connectivity_matrix(laplacian):
    n = laplacian.shape[0]
    return np.multiply(laplacian, -(np.ones_like(laplacian) - 2*np.eye(n)))


# -

# Compute the barycenter of all controls
pseudo_inverses = [scp.linalg.pinv(get_stabilized_laplacian_matrix(connectivity_matrices[c], ratio=1.0)) for c in controls]
barycenter_cov = barycenter_iteration(pseudo_inverses)

plot_connectivity_matrix(connectivity_matrices[controls[9]],laplacian_to_connectivity_matrix(cov_to_laplacian(barycenter_cov)) - np.min(laplacian_to_connectivity_matrix(cov_to_laplacian(barycenter_cov))))

plot_connectivity_matrix(connectivity_matrices[controls[9]],np.clip(laplacian_to_connectivity_matrix(cov_to_laplacian(barycenter_cov)), 0, None))

for i in range(8):
    m = connectivity_matrices[controls[i]]
    with sns.axes_style("white"):
        plt.imshow(np.log(m + 1),cmap='viridis')
        plt.axis('off')
        plt.savefig('control_{}.svg'.format(i))

m = np.clip(laplacian_to_connectivity_matrix(cov_to_laplacian(barycenter_cov)), 0, None)
with sns.axes_style("white"):
    plt.imshow(np.log(m + 1),cmap='viridis')
    plt.axis('off')
    plt.savefig('barycenter.svg')


def subsample_controls_full(controls, sample_size, ratio = 1.0):
    # number of controls used to compute the barycenter
    #controls_barycenter_count = 25
    controls_count = len(controls)

    random_controls_sample = random.sample(range(0, controls_count),sample_size)
    controls_barycenter = np.array(controls)[random_controls_sample]

    pseudo_inverses = [scp.linalg.pinv(get_stabilized_laplacian_matrix(connectivity_matrices[subject],ratio=ratio)) for subject in controls_barycenter]

    return (controls_barycenter, pseudo_inverses)


def do_stuff_full():
    random_fold_count = 0

    control_sample_size = 10

    pickle_folder = "new_new_pickle_data_folding/"

    for ratio in [1.0]:#[0.1,0.15,0.2,0.25,0.3,0.35,0.40,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8]:
        true_folds = 0
        print("Folding with ratio {:.2f}".format(ratio))

        pickle_file_path = pickle_folder + "dist_and_count_full_r{:.2f}_k{}.pickle".format(ratio, control_sample_size)
        print("Will save or update {}".format(pickle_file_path))

        distance_dict = {s : 0.0 for s in patients + controls}
        distance_fold_count = {s:0 for s in patients + controls}

        for fold in range(random_fold_count):
            if fold % 10 == 0: print("Fold {}".format(fold))

            controls_sample, pinvs = subsample_controls_full(controls, control_sample_size, ratio = ratio)

            barycenter_cov = barycenter_iteration(pinvs)

            subjects = patients + [c for c in controls if c not in controls_sample]
            current_fold_distance_dict = {subject:wasserstein_network_covariance(connectivity_matrices[subject], barycenter_cov, ratio = ratio) for subject in subjects}

            for subject, d in current_fold_distance_dict.items():
                distance_fold_count[subject] += 1
                distance_dict[subject] += d

            true_folds += 1

        if os.path.exists(pickle_file_path):
            with open(pickle_file_path, 'rb') as filehandler:
                loaded_dist_dict, loaded_dist_count, loaded_fold_count = pickle.load(filehandler)

                for s in patients + controls:
                    distance_dict[s] += loaded_dist_dict[s]
                    distance_fold_count[s] += loaded_dist_count[s]

                print("{} ({} folds) will be updated with {} more folds".format(pickle_file_path, loaded_fold_count, random_fold_count))
                true_folds += loaded_fold_count

        with open(pickle_file_path, 'wb') as filehandler:
            print("Saving {}".format(pickle_file_path))
            pickle.dump((distance_dict, distance_fold_count, true_folds), filehandler)

        dist_list = []
        for s in patients:
            dist_list.append(distance_dict[s] / distance_fold_count[s])

        for c in controls:
            dist_list.append(distance_dict[c] / distance_fold_count[c])  

        print(scp.stats.ttest_ind(dist_list[:patients_count], dist_list[patients_count:], equal_var=True))
        
        threshold = 0.0025
        print("Plot thresholded at {}".format(threshold))
        plt.hist(np.clip(dist_list[:patients_count],None, threshold), stacked=True, density=True, range=(0,threshold),bins=25, alpha=0.3, label='Patients')
        plt.hist(np.clip(dist_list[patients_count:],None, threshold), stacked=True, density=True, range=(0,threshold), bins=25, alpha = 0.3, label='Controls')
        sns.kdeplot(np.clip(dist_list[:patients_count],None,threshold), shade=False, color='C0', alpha=0.3);
        sns.kdeplot(np.clip(dist_list[patients_count:],None,threshold), shade=False, color='C1', alpha=0.3);
        plt.xlabel("Distance to barycenter ({:.2f})".format(ratio))
        plt.legend()
        import tikzplotlib
        tikzplotlib.save("figures_generation/dist_to_barycenter_full.tex".format(ratio), wrap=False)

        plt.show()


do_stuff_full()

controls

controls_sample, pinvs = subsample_controls_full(controls, 10, ratio = 0.45)
barycenter_cov = barycenter_iteration(pinvs)
barycenter_cov

# load madrs stuff
patient_madrs = {}
with open('data/data_madrs.csv','r')as csv_file:
    data = csv.DictReader(csv_file)
    print(data.fieldnames)
    for row in data:
        patient_madrs[row['Subject']] = row['MADRS ']

ratio = 1.0
with open("new_new_pickle_data_folding/dist_and_count_r{:.2f}_k10.pickle".format(ratio), 'rb') as file:
    distance_dict, distance_fold_count, true_folds = pickle.load(file)
    #plt.plot(distance_unified)
    Y = []
    X = []
    Y_cont = []
    X_cont = []
    feature='MADRS'
    for patient, stats in patient_info_dict.items():
        if patient in distance_dict:
            y = distance_dict[patient] / distance_fold_count[patient]
            if patient in patient_madrs:
                length = int(patient_madrs[patient])#int(snttantts['Duree_maladie'])
            else:
                length = -5
            #lenght = 1.
            #if dist < 1e100:
            X.append(length)
            Y.append(y)
            #plt.plot(dist)

    for control in controls:
        y = distance_dict[control] / distance_fold_count[control]
        X_cont.append(0.)
        Y_cont.append(y)


    #plt.xscale("log", nonposx='clip')
    plt.xlabel(feature)
    plt.ylabel("Distance to barycenter")
    #plt.yscale("log", nonposy='clip')   
    plt.scatter(X,Y,alpha=0.7)
    plt.scatter(X_cont,Y_cont, alpha=0.7)
    
    #import tikzplotlib
    #tikzplotlib.save("figures_generation/vsmadrs_fold_mean_alpha_{:.2f}.tex".format(ratio), wrap=False)
    plt.savefig('no_correl.svg')
    plt.show()



dist_list

threshold = 1500
plt.hist(np.clip(dist_list[:patients_count],None, threshold), stacked=True, density=True, range=(0,threshold),bins=30, alpha=0.3, label='Patients')
plt.hist(np.clip(dist_list[patients_count:],None, threshold), stacked=True, density=True, range=(0,threshold), bins=30, alpha = 0.3, label='Controls')
sns.kdeplot(np.clip(dist_list[:patients_count],None,threshold), shade=False, color='C0', alpha=0.3);
sns.kdeplot(np.clip(dist_list[patients_count:],None,threshold), shade=False, color='C1', alpha=0.3);
plt.xlabel("Distance to barycenter")
plt.legend()
import tikzplotlib
#tikzplotlib.save("barycenter_fold_mean_3.tex", wrap=False)

plt.plot([distance_fold_count[s] for s in patients + controls])

scp.stats.ttest_ind(dist_list[:patients_count], dist_list[patients_count:], equal_var=True)

# ### K-means

# +
k = 2
subjects = patients + controls

update = True

centroids = [None] * k
clusters = [[] for i in range(k)]
subject_cluster_id = {}

ratio = 0.2

# init: random partitions
for s in subjects:
    new_cluster = np.random.randint(k)
    subject_cluster_id[s] = new_cluster
    clusters[new_cluster].append(s)

while update > 0:
    update = 0
    
    subjects_dists = np.zeros((subject_count,k))
    for i in range(k):
        pseudo_inverses_first_hemispher_cluster = [scp.linalg.pinv(get_stabilized_laplacian_matrix(first_hemispher_dict[subject], ratio = ratio)) for subject in clusters[i]]
        pseudo_inverses_second_hemispher_cluster = [scp.linalg.pinv(get_stabilized_laplacian_matrix(second_hemispher_dict[subject], ratio =ratio)) for subject in clusters[i]]
        
        new_centroid_left = barycenter_iteration(pseudo_inverses_first_hemispher_cluster)
        new_centroid_right = barycenter_iteration(pseudo_inverses_second_hemispher_cluster)
        
        centroids[i] = (new_centroid_left, new_centroid_right) 
        
        print("Computing distances for cluster {}".format(i))
        for j, s in enumerate(subjects):
            subjects_dists[j, i] = wasserstein_network_covariance(first_hemispher_dict[s], new_centroid_left, ratio = ratio) + wasserstein_network_covariance(second_hemispher_dict[s], new_centroid_right, ratio = ratio)
    
    clusters = [[] for i in range(k)]
    
    for j, s in enumerate(subjects):
        new_cluster = np.argmin(subjects_dists[j,:])
        if new_cluster != subject_cluster_id[s]:
            update += 1
        
        clusters[new_cluster].append(s)
        subject_cluster_id[s] = new_cluster
    
    print("{} update(s) performed".format(update))
# -

print("finifini")

clusters

# ### Representative control

# +
representative_control = np.inf*np.ones_like(connectivity_matrices[controls[0]])

n = representative_control.shape[0]
for c in controls:
    for i in range(n):
        for j in range(n):
            representative_control[i,j] = min(connectivity_matrices[c][i,j], representative_control[i,j])
            
plt.imshow(np.log(representative_control+1))
plt.imshow(filter_weights(representative_control, ratio = 0.2))
# -

rep_dist = []
for s in patients + controls:
    rep_dist.append(wasserstein_networks(get_hemispher_matrix(connectivity_matrices[s]),get_hemispher_matrix(representative_control), ratio=0.2))

threshold = 2000
plt.hist(np.clip(rep_dist[:patients_count],None,threshold), alpha=0.3, density=True, stacked=True, label='Patients')
plt.hist(np.clip(rep_dist[patients_count:],None,threshold), alpha=0.3, density=True,stacked=True, label='Controls')
sns.kdeplot(np.clip(rep_dist[:patients_count],None,threshold), shade=False, color='C0', alpha=0.3);
sns.kdeplot(np.clip(rep_dist[patients_count:],None,threshold), shade=False, color='C1', alpha=0.3);
plt.legend()
import tikzplotlib
tikzplotlib.save("significant_control.tex", wrap=False)


