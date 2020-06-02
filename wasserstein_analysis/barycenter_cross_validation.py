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
import matplotlib
from matplotlib import pyplot as plt

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

# Folder in which some computed objects will be saved
pickle_folder = 'cross_validation'

# Control whether we recompute data even if it has been previously 
# computed and pickled
recompute_pickle = False

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

# Eventually create the folder in which data will be saved
if not os.path.exists(pickle_folder):
    os.mkdir(pickle_folder)

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
# Those are dictionaries used to speed up computations, by avoiding multiple 
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

# Some functions to print a barycenter covariance as an approximate connectivity matrix
def cov_to_laplacian(cov, alpha = 1e-2):
    n = cov.shape[0]
    return scp.linalg.pinv(cov) - alpha * np.eye(n)

def laplacian_to_connectivity_matrix(laplacian):
    n = laplacian.shape[0]
    return np.multiply(laplacian, -(np.ones_like(laplacian) - 2*np.eye(n)))
# -

# ## Compute a distance matrix (distance subjects to controls)


# +
distance_matrix = np.zeros((subject_count, controls_count))

pickle_path = pickle_folder + '/distance_matrix.pickle'

if os.path.exists(pickle_path) and not recompute_pickle:
    with open(pickle_path, 'rb') as pickle_file:
        print("Loading the distance matrix from {}".format(pickle_path))
        distance_matrix = pickle.load(pickle_file)
        assert distance_matrix.shape == (subject_count, controls_count), \
            "Pickled distance matrix need to be recomputed, set the recompute_pickle flag to True"

else:
    print("Compute a distance matrix. It might take some time.")
    for subject_id, subject in enumerate(patients + controls):
        if subject_id % 10 == 0:
            print("Computing distance for subject {} / {}".format(subject_id, subject_count))

        for control_id, control in enumerate(controls):
            subject_matrix = connectivity_matrices[subject]
            control_matrix = connectivity_matrices[control]
            distance_matrix[subject_id, control_id] = wasserstein_networks(subject_matrix, control_matrix)
    
    with open(pickle_path, 'wb') as pickle_file:
        print("Saving the distance matrix into {}".format(pickle_path))
        pickle.dump(distance_matrix, pickle_file)
# -

print("Plotting the pairwise distance matrix")
plt.imshow(distance_matrix)
plt.colorbar()
plt.show()

# +
print("Plotting distance distribution, from either the patient or the control cohort TO the control cohort.")

plt.hist(distance_matrix[:patients_count].flatten(), bins=30,alpha=0.5, density=True, stacked=True, label="Patients")
plt.hist(distance_matrix[patients_count:].flatten(), bins=30,alpha=0.5, density=True, stacked=True, label="Controls")
sns.kdeplot(distance_matrix[:patients_count].flatten(), shade=True, color='C0', alpha=0.2);
sns.kdeplot(distance_matrix[patients_count:].flatten(), shade=True, color='C1', alpha=0.2);

plt.legend()
plt.xlabel("Distance to control cohort")
plt.show()
# -
# ## Compute a barycenter with subsample of controls


def subsample_controls(controls, sample_size, ratio = 1.0):
    """
        Return a random draw of sample_size controls, whose name are taken from 
        the controls array, as well as the covariance matrices of their 
        associated distribution. Ratio is used to control the sparsity of the 
        networks (see get_stabilized_laplacian_matrix for its exact effect)
    """
    controls_count = len(controls)

    random_controls_sample = random.sample(range(0, controls_count),sample_size)
    controls_barycenter = np.array(controls)[random_controls_sample]

    pseudo_inverses = [scp.linalg.pinv(get_stabilized_laplacian_matrix(connectivity_matrices[subject],ratio=ratio)) for subject in controls_barycenter]

    return (controls_barycenter, pseudo_inverses)


def do_some_folds(fold_count = 100, control_sample_size = 10, ratios = [1.0]):
    """
        Perform fold_count random folds. Each fold consists in: subsampling the 
        control group to randomly select control_sample_size controls, whose 
        barycenter is computed. Then all remaining controls + each patients are 
        compared to the barycenter. The results of multiple folds are aggregated 
        into distance_dict and distance_fold_count (the average distance for 
        subject s to the barycenters is thus 
        
                        distance_dict[s] / distance_fold_count[s]
        
        ratios can be used to perform tests on multiple sparsity ratios at a 
        time 
    """
    for ratio in ratios:
        true_folds = 0
        print("Folding with sparsity ratio {:.2f}".format(ratio))

        pickle_file_path = pickle_folder + "/dist_and_fold_count_r{:.2f}_k{}.pickle".format(ratio, control_sample_size)
        print("Will save or update {}".format(pickle_file_path))

        distance_dict = {s : 0.0 for s in patients + controls}
        distance_fold_count = {s:0 for s in patients + controls}

        for fold in range(fold_count):
            if fold % 10 == 0: print("Fold {}".format(fold))

            controls_sample, pinvs = subsample_controls(controls, control_sample_size, ratio = ratio)

            barycenter_cov = barycenter_iteration(pinvs)
            
            # list of all the other subjects (non sampled controls + patients)
            subjects = patients + [c for c in controls if c not in controls_sample]
            current_fold_distance_dict = {subject:wasserstein_network_covariance(connectivity_matrices[subject], barycenter_cov, ratio = ratio) for subject in subjects}

            for subject, d in current_fold_distance_dict.items():
                distance_fold_count[subject] += 1
                distance_dict[subject] += d

            true_folds += 1

        # create or update the pickled data
        if os.path.exists(pickle_file_path) and not recompute_pickle:
            with open(pickle_file_path, 'rb') as filehandler:
                loaded_dist_dict, loaded_dist_count, loaded_fold_count = pickle.load(filehandler)

                for s in patients + controls:
                    distance_dict[s] += loaded_dist_dict[s]
                    distance_fold_count[s] += loaded_dist_count[s]

                print("{} ({} folds) will be updated with {} more folds".format(pickle_file_path, loaded_fold_count, fold_count))
                true_folds += loaded_fold_count

        with open(pickle_file_path, 'wb') as filehandler:
            print("Saving {}".format(pickle_file_path))
            pickle.dump((distance_dict, distance_fold_count, true_folds), filehandler)

        # plot the results
        
        # first, obtain the true distance lists from the distance_dict and 
        # distance_fold_count dictionnaries
        dist_list = []
        for s in patients:
            dist_list.append(distance_dict[s] / distance_fold_count[s])

        for c in controls:
            dist_list.append(distance_dict[c] / distance_fold_count[c])  

        #print(scp.stats.ttest_ind(dist_list[:patients_count], dist_list[patients_count:], equal_var=True))
        
        #threshold = 0.0025
        #print("Plot thresholded at {}".format(threshold))
        plt.hist(dist_list[:patients_count], stacked=True, density=True, bins=25, alpha=0.3, label='Patients')
        plt.hist(dist_list[patients_count:], stacked=True, density=True, bins=25, alpha = 0.3, label='Controls')
        sns.kdeplot(dist_list[:patients_count], shade=False, color='C0', alpha=0.3);
        sns.kdeplot(dist_list[patients_count:], shade=False, color='C1', alpha=0.3);
        plt.xlabel("Distance to barycenter ({:.2f})".format(ratio))
        plt.legend()
        #import tikzplotlib
        #tikzplotlib.save("figures_generation/dist_to_barycenter_full.tex".format(ratio), wrap=False)

        plt.show()


do_some_folds()


