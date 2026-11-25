import numpy as np

#datastream config
#Datasteam config
#no of samples
STREAM_LENGTH = 9568
FEATURE_DIM = 4

#?? paper simplifies to b = 1 in derivations?? but implementation supports general b >= 1
BATCH_SIZE = 1
#after normalisation
FEATURE_LOWER_BOUND = 0.0
FEATURE_UPPER_BOUND = 1.0

LABEL_LOWER_BOUND = 0.0
LABEL_UPPER_BOUND = 1.0

#Model Params
#(α in Eq. 3, 9)
MODEL_LEARNING_RATE = 0.01

#covergence threshold (ε in Eq. 4 and Eq. 10)
CONVERGENCE_EPSILON = 1e-8

#max outer attack iterations
MAX_ATTACK_ITERATIONS = 10000

RANDOM_SEED = 42

#Poison Params
#γ = (sum of poisoned samples)/(T*b)
POISONING_RATE = 0.1

#(α in Eq. 11, 17–18 outer loop)
POISONING_LEARNING_RATE = 0.01

#(β in Algorithm 1 line 14)
POISONING_DECAY_FACTOR = 0.05

#max iterations inner bi optimization
MAX_INNER_OPTI_ITER = 500

#max iterations outer bi optimization
MAX_OUTER_OPTI_ITER = 500

#check chatgpt code

# sliding window cache (forgets)


# Cache size (b samples per time slice)
CACHE_SIZE = BATCH_SIZE

# Whether cache strictly overwrites previous slice
CACHE_STRICT_OVERWRITE = True


# Rehearsal buffer


# Maximum size of rehearsal buffer (p in paper)
# Must satisfy: p <= total poisoned samples
REHEARSAL_BUFFER_SIZE = int(STREAM_LENGTH * POISONING_RATE)

# Whether rehearsal buffer stores:
# - feature vectors
# - labels
# - corresponding model parameters snapshot (θ_i)
STORE_THETA_SNAPSHOT = True


#grad cimpute


# Use full recursive gradient (Eq. 14–16)
USE_RECURSIVE_GRADIENT = True

# Whether to store Jacobian matrices (I - α x x^T)
STORE_JACOBIAN_PRODUCTS = True


#Edge device simulation


# Simulated RAM constraint (in MB)
SIMULATED_RAM_MB = 256

# Simulated CPU frequency (GHz)
SIMULATED_CPU_GHZ = 1.2

# Whether to measure runtime
MEASURE_RUNTIME = True

# Init settings

#init model params to 0 or random
INITIALIZE_THETA_RANDOM = False
#enable random init
THETA_INIT_STD = 0.01

# Loss (LOT, MSE)
#lot
TRACK_LOSS_OVER_TIME = True

#mse
TRACK_MAX_MIN_LOSS = True

# Numerical Stability
NUMERICAL_STABILITY_EPS = 1e-12
MAX_GRADIEN_NORM = 1e6

# Safety Check
assert 0 < POISONING_RATE < 0.5, \
    "Poisoning rate must satisfy concealment condition γ < 0.5"

assert REHEARSAL_BUFFER_SIZE <= STREAM_LENGTH, \
    "Rehearsal buffer cannot exceed total stream length"

assert BATCH_SIZE >= 1, \
    "Batch size must be >= 1"

assert FEATURE_UPPER_BOUND > FEATURE_LOWER_BOUND, \
    "Feature bounds invalid"

assert LABEL_UPPER_BOUND > LABEL_LOWER_BOUND, \
    "Label bounds invalid"

np.random.seed(RANDOM_SEED)


