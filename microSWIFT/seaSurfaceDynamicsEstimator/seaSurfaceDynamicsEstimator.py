import numpy as np
from numpy.linalg import inv

def seaSurfaceDynamicsEstimator_linear(measured_state, dt):
    '''
    @edwinrainville

    Estimator Algorithm Overview(Adapted from "Bayesian and Kalman Filters"): 

    Initialization
    1. Initialize the state of the filter using first points measured
    2. Initialize a belief in this state

    Predict the next state
    1. Use system behavior to predict the next time step - assume discretized newtonian motion 
    2. Adjust belief in this prediction

    Update the state with Measurements
    1. Get a measurement of the state

    '''

    ## Initialization
    # Define size of state and measurement vectors
    dim_state = 9
    dim_measurement = measured_state.shape[0]
    dim_time = measured_state.shape[1]

    # Define a state estimate array to be filled with values 
    state_est = np.zeros((dim_state, dim_time))
        
    # Define initial uncertainty covariance
    P = np.eye(dim_state)

    # Define state transition matrix - Linear Newtonian system 
    F = np.eye(dim_state)
    F[0,3] = dt
    F[1,4] = dt
    F[2,5] = dt
    F[3,6] = dt
    F[4,7] = dt
    F[5,8] = dt

    # Define initial process uncertainty
    Q = np.eye(dim_state)

    # Define measurement function and uncertainty
    H = np.eye(dim_measurement) # * THIS WILL NEED TO BE UPDATED IN THE FUTURE  
    R = np.eye(dim_measurement)

    # Define initial state as the first set of measurements
    state_est[:,0] = measured_state[:,0]

    # Kalman Filter
    for k in np.arange(1,dim_time):
        # Define the previous step 
        previous_state = state_est[:,k-1].reshape(dim_state, 1)

        ## Predict State Movement to Prior 
        # x = Fx
        state_prior = np.dot(F, previous_state)
        
        # Compute State Covariance matrix
        # P = FPF' + Q
        P = np.dot(F, np.dot(P, F.T)) + Q

        ## Update state movement from prior with measurement
        # Compute Residual
        # y = z - Hx
        y = measured_state[:,k].reshape(dim_state, 1) - np.dot(H, state_prior)
        
        # S = HPH' + R
        # common subexpression for speed
        PHT = np.dot(P, H.T)
        S = np.dot(H, PHT) + R
        S_inv = inv(S)

        # Compute Kalman Gain
        # K = PH'inv(S)
        K = np.dot(PHT, S_inv)

        # Update next step with Kalman gain
        # x = x + Ky
        state_est_current = state_prior + np.dot(K, y)

        # Save updated state into the state estimate matrix
        state_est[:,k] = np.squeeze(state_est_current)

        # Update state covariance matrix
        # P = (I-KH)P(I-KH)' + KRK'
        # P = np.dot((np.eye(P.shape[0]) - np.dot(K, H)), np.dot(P,(np.eye(P.shape[0]) - np.dot(K, H)).T)) + np.dot(K, np.dot(R, K.T))

        # Save the State Covariance matrix

    return state_est
    