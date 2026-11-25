import time
import numpy as np

#other imports
from config import (
    STREAM_LENGTH,
    FEATURE_DIM,
    BATCH_SIZE,
    MODEL_LEARNING_RATE,
    POISONING_LEARNING_RATE,
    POISONING_DECAY_FACTOR,
    POISONING_RATE,
    REHEARSAL_BUFFER_SIZE,
    CONVERGENCE_EPSILON,
    MAX_ATTACK_ITERATIONS,
    INITIALIZE_THETA_RANDOM,
    THETA_INIT_STD,
    MEASURE_RUNTIME,
    RANDOM_SEED
)

from stream import generate_stream
from linear_regression import LinearRegressionOnline
from online_dpa import OnlineDPA

#init
def initialize_model():
    """
    Initialize model parameters according to config.
    """
    if INITIALIZE_THETA_RANDOM:
        theta_init = np.random.normal(
            0,
            THETA_INIT_STD,
            FEATURE_DIM
        )
    else:
        theta_init = np.zeros(FEATURE_DIM)

    model = LinearRegressionOnline(FEATURE_DIM)
    model.theta = theta_init.copy()

    return model

#main
def main():
    np.random.seed(RANDOM_SEED)
    print("Generating online data stream...")
    X_stream, y_stream = generate_stream(
        STREAM_LENGTH,
        FEATURE_DIM,
        filepath='Folds5x2_pp.xlsx'
    )
    
    #model init
    model = initialize_model()

    print("Initial theta:")
    print(model.theta)
    
    #init attacker
    attacker = OnlineDPA(
        model=model,
        model_lr=MODEL_LEARNING_RATE,
        poison_lr=POISONING_LEARNING_RATE,
        poison_decay=POISONING_DECAY_FACTOR,
        gamma=POISONING_RATE,
        rehearsal_size=REHEARSAL_BUFFER_SIZE,
        epsilon=CONVERGENCE_EPSILON,
        max_iter=MAX_ATTACK_ITERATIONS,
        batch_size=BATCH_SIZE
    )
    
    #run attack
    print("\nStarting Rehearsal-Based Online DPA...\n")

    if MEASURE_RUNTIME:
        start_time = time.perf_counter()

    attack_results = attacker.run(
        X_stream=X_stream,
        y_stream=y_stream
    )

    if MEASURE_RUNTIME:
        end_time = time.perf_counter()
        runtime = end_time - start_time
    else:
        runtime = None
    
    #results+ eval
    print ("\n -------- \n ATTACK FINISHED \n --------")
    print("\nFinal model parameters (theta):")
    print(model.theta)

    print("\nFinal MSE on stream:")
    print(attack_results["final_mse"])

    print("\nTotal poisoned samples:")
    print(attack_results["total_poisoned"])

    print("\nMaximum observed loss:")
    print(attack_results["mse_max"])

    print("\nMinimum observed loss:")
    print(attack_results["mse_min"])

    if runtime is not None:
        print("\nTotal attack runtime (seconds):")
        print(runtime)

    print("\nDone.")


if __name__ == "__main__":
    main()
