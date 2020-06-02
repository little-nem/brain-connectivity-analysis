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

# ## Some demonstrations 

print("Some connectivity matrices: {} and {}".format(controls[0], patients[0]))
plot_connectivity_matrix(connectivity_matrices[controls[0]], connectivity_matrices[patients[0]])

print("Compute the network associated to a subject ({})".format(controls[0]))
G = get_network(connectivity_matrices[controls[0]])
nx.draw(G)
plt.show()

print("Wasserstein distance between two connectivity networks: {}".format(wasserstein_networks(connectivity_matrices[controls[0]], connectivity_matrices[patients[0]])))

# +
barycenter_cohort = 10

print("Computing a barycenter of {} controls.".format(barycenter_cohort))

# select the controls that we will use to compute the barycenter
subset_of_controls = controls[:10]

# compute the covariances of the corresponding distributions
subset_of_connectivity = [connectivity_matrices[control] for control in subset_of_controls]
subset_of_laplacians = [get_stabilized_laplacian_matrix(m) for m in subset_of_connectivity]
covariance_list = [scp.linalg.pinv(L) for L in subset_of_laplacians]

barycenter_covariance = barycenter_iteration(covariance_list)
m = laplacian_to_connectivity_matrix(cov_to_laplacian(barycenter_covariance))

# note that we need to clip, because the barycenter does not necessarily 
# corresponds to a true connectivity matrix, we only show an approximation
print("Plotting the barycenter as an approximated connectivity matrix.")
plot_connectivity_matrix(np.clip(m,0,None))
# -


