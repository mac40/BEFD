library(astsa)
library(fImport)

# Ex IF2
set.seed(420)
theta_hat = matrix(NA,1000,3)
for (i in 1:1000){
  y = arima.sim(list(order = c(1,0,1), ar =.9,ma = .5),n=200, sd = 1)
  fitted_model = arima(y,c(1,0,1))
  phi = fitted_model$coef[1]
  theta = fitted_model$coef[2]
  sigma2 = fitted_model$sigma2
  theta_hat[i,] = c(phi,theta,sigma2)
}

phi = theta_hat[,1]
plot(density(phi),col = 2,main="phi")
hist(phi,probability = T,add = T)

theta = theta_hat[,2]
plot(density(theta),col = 2,main="theta")
hist(theta,probability = T,add = T)

sigma2 = theta_hat[,3]
plot(density(sigma2), col = 2,main="sigma^2")
hist(sigma2,probability = T,add = T)

means = c(mean(phi),mean(theta),mean(sigma2))
sds = c(sd(phi),sd(theta),sd(sigma2))
trueval = c(0.9,0.5,1)
lambda = rep(NA,3)
for (i in 1:3){
  lambda[i] = (means[i]-trueval[i])/sds[i]
}

# Ex P7
plot(globtemp)
glob_diff = diff(globtemp)
plot(glob_diff)
acf2(glob_diff)
mod_ma2 = sarima(glob_diff,0,0,2)
mod_ar3 = sarima(glob_diff,3,0,0)
mod_arma11 = sarima(glob_diff,1,0,1)
AIC = c(mod_ma2$AIC,mod_ar3$AIC,mod_arma11$AIC)
AICc = c(mod_ma2$AICc,mod_ar3$AICc,mod_arma11$AICc)
BIC = c(mod_ma2$BIC,mod_ar3$BIC,mod_arma11$BIC)
mod = arima(glob_diff,c(1,0,1))
pred = predict(mod,10)$`pred`
se = predict(mod,10)$se
lower_bound = pred - 1.96 * se
upper_bound = pred + 1.96 * se

# Ex P9
plot(so2)
plot(log(so2))
acf2(log(so2),100)
sarima(log(so2),2,0,0)
mod = arima(log(so2),c(2,0,0))
pred = predict(mod,4)$`pred`
se = predict(mod,4)$se
lower_bound = pred - 1.96 * se
upper_bound = pred + 1.96 * se