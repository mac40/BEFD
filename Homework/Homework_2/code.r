library("astsa")

###########################################################################
## Example #1. 
# FITLERING AND SMOOTHING DATA USING SIMULATED DATA
# PARAMETERS ARE FIXED
# generate data
set.seed(1)
num = 50
w = rnorm(num+1,0,1); 
v = rnorm(num,0,1)
mu = cumsum(w) 					# state: mu[0], mu[1],..., mu[50]
y = mu[-1] + v 						# obs: y[1],..., y[50]

# filter and smooth (Ksmooth0 does both)
ks = Ksmooth0(num, y, A=1, mu0=0, Sigma0=1, Phi=1, cQ=1, cR=1) 

# start figure
par(mfrow=c(3,1)); 
Time = 1:num
plot(Time, mu[-1], main='Predict', ylim=c(-5,10))
lines(ks$xp)
lines(ks$xp+2*sqrt(ks$Pp), lty=2, col=4)
lines(ks$xp-2*sqrt(ks$Pp), lty=2, col=4)
plot(Time, mu[-1], main='Filter', ylim=c(-5,10)) 
lines(ks$xf)
lines(ks$xf+2*sqrt(ks$Pf), lty=2, col=4) 
lines(ks$xf-2*sqrt(ks$Pf), lty=2, col=4)
plot(Time, mu[-1], main='Smooth', ylim=c(-5,10)) 
lines(ks$xs)
lines(ks$xs+2*sqrt(ks$Ps), lty=2, col=4)
lines(ks$xs-2*sqrt(ks$Ps), lty=2, col=4)

mu[1]
ks$x0n
sqrt(ks$P0n)  # initial value info

###########################################################################
## Example #2. 
# PERFORM MLE ON SIMULATED DATA
# Generate Data
set.seed(999)
num = 100
x = arima.sim(n=num+1, list(ar=.8), sd=1)
y = ts(x[-1] + rnorm(num,0,1))

# Initial Estimates
u    = ts.intersect(y, lag(y,-1), lag(y,-2))
varu = var(u)
coru = cor(u)
phi  = coru[1,3]/coru[1,2]
q    = (1-phi^2)*varu[1,2]/phi
r    = varu[1,1] - q/(1-phi^2)
(init.par = c(phi, sqrt(q), sqrt(r))) # = .91, .51, 1.03 # Function to evaluate the likelihood
Linn = function(para){
  phi = para[1]; sigw = para[2]; sigv = para[3]
  Sigma0 = (sigw^2)/(1-phi^2); Sigma0[Sigma0<0]=0
  kf = Kfilter0(num, y, 1, mu0=0, Sigma0, phi, sigw, sigv)
  return(kf$like)}

# Estimation (partial output shown)
(est = optim(init.par, Linn, gr=NULL, method='BFGS', hessian=TRUE, control=list(trace=1, REPORT=1)))
SE = sqrt(diag(solve(est$hessian)))
cbind(estimate=c(phi=est$par[1],sigw=est$par[2],sigv=est$par[3]),SE)

###########################################################################
## Example #3. 
## GLOBAL TEMPERATURE DEVIATIONS
y = cbind(globtemp, globtempl) 
num = nrow(y) 
input = rep(1,num)
A = array(rep(1,2), dim=c(2,1,num))
mu0 =-.35;Sigma0=1; Phi=1

# Function to Calculate Likelihood
Linn   = function(para) {
  cQ = para[1]			# sigma_w
  cR1 = para[2] 			# 11 element of chol(R)
  cR2 = para[3]			# 22 element of chol(R)
  cR12 = para[4]			# 12 element of chol(R)
  
  cR    = matrix(c(cR1,0,cR12,cR2),2)  # put the matrix together
  drift = para[5]
  kf    = Kfilter1(num,y,A,mu0,Sigma0,Phi,drift,0,cQ,cR,input)
  return(kf$like)         
}

# Estimation
init.par = c(.1,.1,.1,0,.05) # initial values of parameters 
(est = optim(init.par, Linn, NULL, method='BFGS', hessian=TRUE,
             control=list(trace=1,REPORT=1)))  # output not shown
SE = sqrt(diag(solve(est$hessian)))

# Display estimates
u = cbind(estimate=est$par, SE) 
rownames(u)=c('sigw','cR11', 'cR22', 'cR12', 'drift')
u 

# Smooth (first set parameters to their final estimates) 
cQ = est$par[1]
cR1  = est$par[2]
cR2  = est$par[3]
cR12 = est$par[4]
cR = matrix(c(cR1,0,cR12,cR2), 2)
(R = t(cR)%*%cR) # to view the estimated R matrix 
drift = est$par[5]
ks = Ksmooth1(num,y,A,mu0,Sigma0,Phi,drift,0,cQ,cR,input) 

# Plot
xsm = ts(as.vector(ks$xs), start=1880)
rmse = ts(sqrt(as.vector(ks$Ps)), start=1880)
plot(xsm, ylim=c(-.6, 1), ylab='Temperature Deviations')
xx = c(time(xsm), rev(time(xsm)))
yy = c(xsm-2*rmse, rev(xsm+2*rmse))
polygon(xx, yy, border=NA, col=gray(.6, alpha=.25))
lines(globtemp, type='o', pch=2, col=4, lty=6) 
lines(globtempl, type='o', pch=3, col=3, lty=6)


###########################################################################
## Example #4. 
## Expectation-Maximization algorithm
library(nlme) # loads package nlme
# Generate data 
set.seed(999); 
num = 100
x = arima.sim(n=num+1, list(ar = .8), sd=1) 
y = ts(x[-1] + rnorm(num,0,1))

# Initial Estimates 
# Initial Estimates 
u = ts.intersect(y, lag(y,-1), lag(y,-2)) 
varu = var(u); coru = cor(u)
phi = coru[1,3]/coru[1,2]
q= (1-phi^2)*varu[1,2]/phi
r= varu[1,1] - q/(1-phi^2)

# EM procedure - output not shown
(em = EM0(num, y, A=1, mu0=0, Sigma0=2.8, Phi=phi, cQ=sqrt(q), cR=sqrt(r),
          max.iter=75, tol=.00001))
# Standard Errors  (this uses nlme)
phi = em$Phi; cq = sqrt(em$Q); cr = sqrt(em$R)
mu0 = em$mu0; Sigma0 = em$Sigma0
para = c(phi, cq, cr)
Linn = function(para){ # to evaluate likelihood at estimates
  kf = Kfilter0(num, y, 1, mu0, Sigma0, para[1], para[2], para[3])
  return(kf$like)       }
emhess = fdHess(para, function(para) Linn(para))
SE     = sqrt(diag(solve(emhess$Hessian)))

# Display Summary of Estimation
estimate = c(para, em$mu0, em$Sigma0); SE = c(SE, NA, NA) 
u = cbind(estimate, SE)
rownames(u) = c('phi','sigw','sigv','mu0','Sigma0'); u

###########################################################################
## Example #5. 
## LONGITUDIAL BIOMEDICAL DATA
y = cbind(WBC, PLT, HCT)
num = nrow(y)

# make array of obs matrices
A = array(0, dim=c(3,3,num))
for(k in 1:num) { if (y[k,1] > 0) A[,,k]= diag(1,3) } 
# Initial values
mu0 = matrix(0, 3, 1)
Sigma0 = diag(c(.1, .1, 1), 3)
Phi = diag(1, 3)
cQ = diag(c(.1, .1, 1), 3); 
cR = diag(c(.1, .1, 1), 3) 

# EM procedure - some output previously shown
(em = EM1(num, y, A, mu0, Sigma0, Phi, cQ, cR, 100, .001))

# Graph smoother
ks = Ksmooth1(num, y, A, em$mu0, em$Sigma0, em$Phi, 0, 0, chol(em$Q),
              chol(em$R), 0)
y1s = ks$xs[1,,]; y2s = ks$xs[2,,]; y3s = ks$xs[3,,]
p1 = 2*sqrt(ks$Ps[1,1,])
p2 = 2*sqrt(ks$Ps[2,2,])
p3 = 2*sqrt(ks$Ps[3,3,]) 
par(mfrow=c(3,1))
plot(WBC, type='p', pch=19, ylim=c(1,5), xlab='day')
lines(y1s)
lines(y1s+p1, lty=2, col=4)
lines(y1s-p1, lty=2, col=4) 

plot(PLT, type='p', ylim=c(3,6), pch=19, xlab='day')
lines(y2s)
lines(y2s+p2, lty=2, col=4)
lines(y2s-p2, lty=2, col=4) 

plot(HCT, type='p', pch=19, ylim=c(20,40), xlab='day')
lines(y3s)
lines(y3s+p3, lty=2, col=4)
lines(y3s-p3, lty=2, col=4)



###########################################################################
## Example #6.
## JOHNSON & JOHNSON QUARTERLY EARNINGS
num = length(jj)
A = cbind(1,1,0,0)

# Function to Calculate Likelihood 
Linn =function(para){
  Phi = diag(0,4)
  Phi[1,1] = para[1]
  Phi[2,]=c(0,-1,-1,-1)
  Phi[3,]=c(0,1,0,0)
  Phi[4,]=c(0,0,1,0)
  cQ1 = para[2] 
  cQ2 = para[3]      # sqrt q11 and q22
  cQ  = diag(0,4) 
  cQ[1,1]=cQ1
  cQ[2,2]=cQ2
  cR  = para[4]                     # sqrt r11
  kf  = Kfilter0(num, jj, A, mu0, Sigma0, Phi, cQ, cR)
  return(kf$like)  
}

# Initial Parameters
mu0 = c(.7,0,0,0)
Sigma0 = diag(.04,4)
init.par = c(1.03,.1,.1,.5) # Phi[1,1], the 2 cQs and cR # Estimation and Results

est = optim(init.par, Linn,NULL, method='BFGS', hessian=TRUE,control=list(trace=1,REPORT=1))
SE  = sqrt(diag(solve(est$hessian)))
u = cbind(estimate=est$par, SE) 
rownames(u)=c('Phi11','sigw1','sigw2','sigv') 
u

# Smooth
Phi = diag(0,4)
Phi[1,1] = est$par[1]
Phi[2,]=c(0,-1,-1,-1)
Phi[3,]=c(0,1,0,0) 
Phi[4,]=c(0,0,1,0) 
cQ1 = est$par[2] 
cQ2 = est$par[3]
cQ = diag(1,4) 
cQ[1,1]=cQ1
cQ[2,2]=cQ2
cR = est$par[4]
ks = Ksmooth0(num, jj, A, mu0, Sigma0, Phi, cQ, cR)

# Plots
Tsm = ts(ks$xs[1,,], start=1960, freq=4)
Ssm = ts(ks$xs[2,,], start=1960, freq=4)
p1 = 3*sqrt(ks$Ps[1,1,])
p2 = 3*sqrt(ks$Ps[2,2,]) 
par(mfrow=c(2,1))
plot(Tsm, main='Trend Component', ylab='Trend')
xx = c(time(jj), rev(time(jj)))
yy = c(Tsm-p1, rev(Tsm+p1))
polygon(xx, yy, border=NA, col=gray(.5, alpha = .3))
plot(jj, main='Data & Trend+Season', ylab='J&J QE/Share', ylim=c(-.5,17))
xx = c(time(jj), rev(time(jj)) )
yy = c((Tsm+Ssm)-(p1+p2), rev((Tsm+Ssm)+(p1+p2)) )
polygon(xx, yy, border=NA, col=gray(.5, alpha = .3))

# Forecast
n.ahead = 12
y = ts(append(jj, rep(0,n.ahead)), start=1960, freq=4)
rmspe = rep(0,n.ahead)
x00 = ks$xf[,,num]
P00 = ks$Pf[,,num]
Q = t(cQ)%*%cQ
R = t(cR)%*%(cR)
for (m in 1:n.ahead){
  xp  = Phi%*%x00
  Pp = Phi%*%P00%*%t(Phi)+Q
  sig = A%*%Pp%*%t(A)+R
  K = Pp%*%t(A)%*%(1/sig)
  x00 = xp; P00 = Pp-K%*%A%*%Pp
  y[num+m] = A%*%xp; rmspe[m] = sqrt(sig)  
}

plot(y, type='o', main='', ylab='J&J QE/Share', ylim=c(5,30), xlim=c(1975,1984))
upp = ts(y[(num+1):(num+n.ahead)]+2*rmspe, start=1981, freq=4)
low = ts(y[(num+1):(num+n.ahead)]-2*rmspe, start=1981, freq=4)
xx = c(time(low), rev(time(upp)))
yy = c(low, rev(upp))
polygon(xx, yy, border=8, col=gray(.5, alpha = .3))
abline(v=1981, lty=3)

###########################################################################
## Example #7.
## Stochastic regression: inflation vs interest rates
library(plyr) # used for displaying progress
tol = sqrt(.Machine$double.eps) # determines convergence of optimizer nboot = 500 # number of bootstrap replicates
y     = window(qinfl, c(1953,1), c(1965,2))  # inflation
z     = window(qintr, c(1953,1), c(1965,2))  # interest
num = length(y)
A = array(z, dim=c(1,1,num))
input = matrix(1,num,1)

# Function to Calculate Likelihood
Linn = function(para, y.data){ # pass data also
  phi = para[1]; alpha = para[2]
  b   = para[3]; Ups   = (1-phi)*b
  cQ  = para[4]; cR    = para[5]
  kf  = Kfilter2(num,y.data,A,mu0,Sigma0,phi,Ups,alpha,1,cQ,cR,0,input)
  return(kf$like)    
}

# Parameter Estimation
mu0 = 1; Sigma0 = .01
init.par = c(phi=.84, alpha=-.77, b=.85, cQ=.12, cR=1.1) # initial values
est = optim(init.par,  Linn, NULL, y.data=y, method="BFGS", hessian=TRUE,
            control=list(trace=1, REPORT=1, reltol=tol))
SE  = sqrt(diag(solve(est$hessian)))
phi = est$par[1]; alpha = est$par[2]
b   = est$par[3]; Ups   = (1-phi)*b
cQ  = est$par[4]; cR    = est$par[5]
round(cbind(estimate=est$par, SE), 3)

# Run the filter at the estimates
kf = Kfilter1(num,y,A,mu0,Sigma0,phi,Ups,alpha,1,cQ,cR)
plot(kf$xp, type="l", ylim=c(-1.20,2.0))
lines(kf$xp+2*sqrt(kf$Pf), lty=2, col=4) 
lines(kf$xp-2*sqrt(kf$Pf), lty=2, col=4)

# Run the smoother at the estimates
ks = Ksmooth1(num,y,A,mu0,Sigma0,phi,Ups,alpha,1,cQ,cR)
plot(ks$xs, type="l", ylim=c(-1.30,2.20))
lines(ks$xf+2*sqrt(ks$Pf), lty=2, col=4) 
lines(ks$xf-2*sqrt(ks$Pf), lty=2, col=4)

###########################################################################
## Example #8.
## Number of Major Heartquakes
install.packages("depmixS4")
library("depmixS4")
library("nnet")
library("MASS")
library("Rsolnp")
model <- depmix(EQcount ~1, nstates=2, data=data.frame(EQcount),
                family=poisson())
set.seed(90210)
summary(fm <- fit(model)) # estimation results

##-- Get Parameters --##
u = as.vector(getpars(fm)) # ensure state 1 has smaller lambda
if (u[7] <= u[8]) { para.mle = c(u[3:6], exp(u[7]), exp(u[8])) } else { para.mle = c(u[6:3], exp(u[8]), exp(u[7])) }
mtrans = matrix(para.mle[1:4], byrow=TRUE, nrow=2)
lams = para.mle[5:6]
pi1 = mtrans[2,1]/(2 - mtrans[1,1] - mtrans[2,2]); pi2 = 1-pi1

##-- Graphics --##
layout(matrix(c(1,2,1,3), 2))
par(mar = c(3,3,1,1), mgp = c(1.6,.6,0))

# data and states
plot(EQcount, main="", ylab="EQcount", type="h", col=gray(.7))
text(EQcount, col=6*posterior(fm)[,1]-2, labels=posterior(fm)[,1], cex=.9) 

# prob of state 2
plot(ts(posterior(fm)[,3], start=1900), ylab = expression(hat(pi)[~2]*'(t|n)')); abline(h=.5, lty=2)

# histogram
hist(EQcount, breaks=30, prob=TRUE, main="")
xvals = seq(1,45)
u1 = pi1*dpois(xvals, lams[1])
u2 = pi2*dpois(xvals, lams[2])
lines(xvals, u1, col=4); lines(xvals, u2, col=2)

##-- Bootstap --##
# function to generate data
pois.HMM.generate_sample = function(n,m,lambda,Mtrans,StatDist=NULL){
  # n = data length, m = number of states, Mtrans = transition matrix, StatDist = stationary distn
  if(is.null(StatDist)) StatDist = solve(t(diag(m)-Mtrans +1),rep(1,m))
  mvect = 1:m
  state = numeric(n)
  state[1] = sample(mvect ,1, prob=StatDist)
  for (i in 2:n)
    state[i] = sample(mvect ,1,prob=Mtrans[state[i-1] ,])
  y = rpois(n,lambda=lambda[state ])
  list(y= y, state= state)
}

# start it up
set.seed(10101101)
nboot = 100
nobs = length(EQcount)
para.star = matrix(NA, nrow=nboot, ncol = 6)
for (j in 1:nboot){
  x.star = pois.HMM.generate_sample(n=nobs, m=2, lambda=lams, Mtrans=mtrans)$y
  model <- depmix(x.star ~1, nstates=2, data=data.frame(x.star),
                  family=poisson())
  u = as.vector(getpars(fit(model, verbose=0)))
  # make sure state 1 is the one with the smaller intensity parameter
  if (u[7] <= u[8]) { para.star[j,] = c(u[3:6], exp(u[7]), exp(u[8])) }
  else  { para.star[j,] = c(u[6:3], exp(u[8]), exp(u[7])) }     
}

# bootstrapped std errors
SE = sqrt(apply(para.star,2,var) +
            (apply(para.star,2,mean)-para.mle)^2)[c(1,4:6)]
names(SE)=c('seM11/M12', 'seM21/M22', 'seLam1', 'seLam2'); SE