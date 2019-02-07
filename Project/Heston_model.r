NMLE-CEKF = function(S, T_ = length(S), V0 = 0.2, r = 0.005, kaph0.hat = 1, theta0.hat = 0.250, sigma0.hat = 0.5, delta = 0.01){
  # INPUT:
  # k = time of execution
  # todo: describe input variables
  
  ro0.hat = 0.000001 # temporarily initialized to a random low value
                     # need to find proper initialization
  
  
  
  #VARIABLES
  
  # Initialization parameters (at time k = 0)
  V0.hat = var(S[1:5])
  P0 = mean((S[1:5]-mean(S[1:5]))^4)
  
  # from function parameters
  # kaph0.hat
  # theta0.hat
  # sigma0.hat
  ro = ro0.hat

  # additional variables
  # time/measurements ratio
  deltaT = T_/n
  # Generating S0 as the mean of future stock value (random value required)
  S0 = mean(S[1:5])
  # Dimension of stock value as number of iterations
  n = length(S)

  # deltaQ = matrix(nrow = n+1) tbd
  # deltaR = matrix(nrow = n) tbd
  
  V.hat_ = matix(nrow = n)
  F_ = matrix(nrow = n)
  L_ = matrix(nrow = n)
  V.bar_ = matrix(nrow = n)
  P.bar_ = matrix(nrow = n)
  H_ = matrix(nrow = n)
  M_ = matrix(nrow = n)
  K_ = matrix(nrow = n)
  P_ = matrix(nrow = n)
  Q_ = matrix(nrow = n+1)
  kaph.hat = matrix(nrow = n)
  theta.hat = matrix(nrow = n)
  sigma.hat = matrix(nrow = n)
  zeta_ = matrix(nrow = n+1); zeta_[1] = log(S[1]) - log(S0)
  for(i in 2:n+1){
    zeta_[i] = log(S[i+1]) - log(S[i])
  }
  
  # first iteration with initial values
  F0 = 1 - kaph0.hat %*% deltaT
  L0 = c(0, sigma0.hat %*% sqrt(V.hat_))
  V.bar_[1] = V0.hat + kaph0.hat %*% theta0.hat %*% deltaT - kaph0.hat %*% V0.hat %*% deltaT
  P.bar_[1] = F0 %*% P0 %*% t(F0) + L0 %*% Q_[1] %*% t(L0) + deltaQ0
  H_[1] = -1/2 * deltaT
  M_[1] = c(sqrt((1-ro0.hat) %*% V.bar_[1] %*% deltaT),ro0.hat %*% sqrt(V.bar_[1] %*% deltaT))
  K_[1] = (P.bar_[1] %*% t(H_[1])
           + L0 %*% Q[1] %*% t(M_[1]))
          %*%
          solve(H_[1] %*% P0 %*% t(H_[1])
                + M_[1] %*% Q_[1] %*% t(M_[1])
                + H_[1] %*% L0 %*% Q_[1] %*% t(M_[1])
                + M_[1] %*% Q_[1] %*% t(L0) %*% t(H_[1]))
  V.hat_[1] = V.bar_[1] + K_[1] %*% (zeta_[1] - 1/2 %*% V.bar_[1])
  P_[1] = P.bar_[1] - K_[1] %*% ( H_[1] %*% P.bar_[1] + M_[1] %*% t(L0)) + deltaR[1]
  
  for(k in 1:(n-1)){
    F_[k] = 1-kaph0.hat_ %*% deltaT
    L_[k] = c(0, sigma0.hat %*% sqrt(V.hat_))
    V.bar_[k+1] = V.hat_[k] + kaph0.hat %*% theta0.hat %*% deltaT - kaph0.hat %*% V.hat_[k] %*% deltaT
    P.bar_[k+1] = F_[k] %*% P_[k] %*% t(F_[k]) + L_[k] %*% Q_[k+1] %*% t(L_[k]) + deltaQ[k]
    H_[k+1] = -1/2 * deltaT
    M_[k+1] = c(sqrt((1-ro0.hat) %*% V.bar_[k+1] %*% deltaT),ro0.hat %*% sqrt(V.bar_[k+1] %*% deltaT))
    K_[k+1] = (P.bar_[k+1] %*% t(H_[k+1])
             + L_[k] %*% Q[k+1] %*% t(M_[k+1]))
    %*%
      solve(H_[k+1] %*% P.bar_[k] %*% t(H_[k+1])
            + M_[k+1] %*% Q_[k+1] %*% t(M_[k+1])
            + H_[k+1] %*% L_[k] %*% Q_[k+1] %*% t(M_[k+1])
            + M_[k+1] %*% Q_[k+1] %*% t(L_[k]) %*% t(H_[k+1]))
    V.hat_[k+1] = V.bar_[k+1] + K_[k+1] %*% (zeta_[k+1] - 1/2 %*% V.bar_[k+1])
    P_[k+1] = P.bar_[k+1] - K_[k+1] %*% ( H_[k+1] %*% P.bar_[k+1] + M_[k+1] %*% t(L_[k])) + deltaR[k+1]
  }
}