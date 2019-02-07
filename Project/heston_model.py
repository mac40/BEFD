'''
Heston model algorithm
'''
# pylint: disable-msg=C0103

import numpy as np


def nmle_cekf(S):
    '''
    Normal Maximium Likelihood Estimation Consistent Extended Kalman Filter function
    '''
    # VARIABLES

    #   EXTENDED KALMAN FILTER
    V_HAT = np.empty(len(S)) # Predicted Volatility
    V_BAR = np.empty(len(S)) # State prediction estimation
    P = np.empty(len(S)) # Variance Covariance matrix upper-bound
    P_BAR = np.empty(len(S)) # Prediction estimation-error covariance
    Q = [None]*len(S) # list of 2x2 matrices
    K = np.empty(len(S)) # Kalman Gain
    #       LINEARIZATION MATRICES
    #           STATE FUNCTION
    F = np.empty(len(S))
    L = [None]*len(S) # list of 2x1 matrices
    #           MEASUREMENT FUNCTION
    H = np.empty(len(S))
    M = [None]*len(S) # list of 2x1 matrices

    #   MAXIMUM LIKELIHOOD ESTIMATION
    KAPTH_HAT = np.empty(len(S))
    THETA_HAT = np.empty(len(S))
    SIGMA_HAT = np.empty(len(S))
    RO_HAT = np.empty(len(S))

    # AUXILIARY VARIABLES
    ZETA = np.empty(len(S))
    for i, _ in enumerate(S):
        ZETA[i] = np.log(S[i+1]) - np.log(S[i])
    deltaT = 1/250 # year/measurements_per_year ration
    R = 0.0005 # Annual interest rate
    deltaQ = np.empty(len(S)) # tbd
    deltaR = np.empty(len(S)) # tbd
    deltaW1 = np.empty(len(S))
    deltaW2 = np.empty(len(S))
    delta = 0.01

    # INITIALIZATION
    S = np.concatenate([np.mean(S[:5])], S) # new S[0]
    V_HAT[0] = np.var(np.log(S[1:5]/S[:4]))
    P[0] = np.mean((S[:5]-np.mean(S[:5]))**4)

    KAPTH_HAT[0] = 1
    THETA_HAT[0] = 0.250
    SIGMA_HAT[0] = 0.5
    RO_HAT[0] = 0.0001

    #UPDATE
    for k, _ in enumerate(S):
        # LINEARIZATION MATRICES OF THE STATE FUNCTION
        F[k] = 1 - KAPTH_HAT[k] * deltaT
        L[k] = np.array([0, SIGMA_HAT[k] * np.sqrt(V_HAT[k] * deltaT)])
        # STATE PREDICTION ESTIMATION AND PREDICTION ESTIMATION-ERROR COVARIANCE
        V_BAR[k+1] = V_HAT[k] \
                    + KAPTH_HAT[k] * SIGMA_HAT[k] * deltaT \
                    - KAPTH_HAT[k] * V_HAT[k] * deltaT
        P_BAR[k+1] = F[k] * P[k] * np.transpose(F[k]) \
                    + L[k] * Q[k] * np.transpose(L[k]) \
                    + deltaQ[k]
        # LINEARIZATION MATRICES OF THE MEASUREMENT FUNCTION
        H[k+1] = -1/2 * deltaT
        M[k+1] = np.array([np.sqrt((1 - RO_HAT[k]**2) * V_BAR[k+1] * deltaT),
                           RO_HAT[k] * np.sqrt(V_BAR[k+1] * deltaT)])
        # STATE ESTIMATE AND ERROR COVARIANCE
        K[k+1] = (P_BAR[k+1] * np.transpose(H[k+1]) \
                  + L[k] * Q[k] * np.transpose(M[k+1])) \
                 * \
                 (H[k+1] * P_BAR[k] * np.transpose(H[k+1]) \
                  + M[k+1] * Q[k] * np.transpose(M[k+1]) \
                  + H[k+1] * L[k] * Q[k] * np.transpose(M[k+1]) \
                  + M[k+1] * Q[k] * np.transpose(L[k]) * np.transpose(H[k+1]))**(-1)
        V_HAT[k+1] = V_BAR[k+1] + K[k+1] * (ZETA[k+1] - 1/2 * V_BAR[k+1])
        P[k+1] = P_BAR[k+1] \
                - K[k+1] \
                * (H[k+1] * P_BAR[k+1] + M[K+1] * np.transpose(L[k])) \
                + deltaR[k+1]
        # PARAMETER ESTIMATION
        P_HAT = (1/k * np.sum(np.sqrt(V_HAT[1:k+1] * V_HAT[:k])) \
                 - 1/(k**2) * np.sum(np.sqrt(V_HAT[1:k+1]/V_HAT[:k])) * np.sum(V_HAT[k])) \
                * \
                (delta/2 \
                 - delta/2 * 1/(k**2) * np.sum(1/V_HAT[k]) * np.sum(V_HAT[k]))**(-1)
        KAPTH_HAT[k+1] = 2 * delta**(-1) * (1 + (P_HAT * delta)/2 * 1/k * np.sum(1/V_HAT[:k]) \
                                       - 1/k * np.sum(np.sqrt(V_HAT[1:k+1]/V_HAT[:k])))
        SIGMA_HAT[k+1] = np.sqrt(4/delta * 1/k * np.sum((np.sqrt(V_HAT[1:k+1]) \
                         - np.sqrt(V_HAT[:k]) \
                         - delta/(2 * np.sqrt(V_HAT[:k])) \
                         * (P_HAT - KAPTH_HAT[k+1] * V_HAT[:k]))**2))
        THETA_HAT[k+1] = (P_HAT + 1/4 * SIGMA_HAT[k+1]**2)/KAPTH_HAT[k+1]

        deltaW1[k+1] = (np.log(S[k+1]) - np.log(S[k]) - (R - 1/2 * V_HAT[k]) * delta) \
                      /(np.sqrt(V_HAT[k]))
        deltaW2[k+1] = (V_HAT[k+1] - V_HAT[k] - KAPTH_HAT[k+1] \
                      * (THETA_HAT[k+1] -V_HAT[k]) * delta) \
                      / (SIGMA_HAT[k+1] * np.sqrt(V_HAT[k]))
        RO_HAT[k+1] = 1/(k * delta) * np.sum(deltaW1[1:k+1] * deltaW2[1:k+1])
    return V_HAT

if __name__ == "__main__":
    pass
