library(astsa)
library(readxl)
library(gamlss.data)
library (depmixS4)

## Exercise P8

num = length(unemp)
A = cbind(1,1,0,0,0,0,0,0,0,0,0,0)

Linn =function(para){
  Phi = diag(0,12)
  Phi[1,1] = para[1]
  Phi[2,]=c(0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1)
  Phi[3,]=c(0,1,0,0,0,0,0,0,0,0,0,0)
  Phi[4,]=c(0,0,1,0,0,0,0,0,0,0,0,0)
  Phi[5,]=c(0,0,0,1,0,0,0,0,0,0,0,0)
  Phi[6,]=c(0,0,0,0,1,0,0,0,0,0,0,0)
  Phi[7,]=c(0,0,0,0,0,1,0,0,0,0,0,0)
  Phi[8,]=c(0,0,0,0,0,0,1,0,0,0,0,0)
  Phi[9,]=c(0,0,0,0,0,0,0,1,0,0,0,0)
  Phi[10,]=c(0,0,0,0,0,0,0,0,1,0,0,0)
  Phi[11,]=c(0,0,0,0,0,0,0,0,0,1,0,0)
  Phi[12,]=c(0,0,0,0,0,0,0,0,0,0,1,0)
  cQ1 = para[2]
  cQ2 = para[3]
  cQ  = diag(0,12)
  cQ[1,1]=cQ1
  cQ[2,2]=cQ2
  cR  = para[4]
  kf  = Kfilter0(num, unemp, A, mu0, Sigma0, Phi, cQ, cR)
  return(kf$like)
}

mu0 = rep(0,12); mu0[1] = unemp[1]
Sigma0 = diag(sd(unemp[1:12])/12,12)
init.par = c(1.03,.1,.1,.5)

est = optim(init.par, Linn,NULL, method='L-BFGS-B', hessian=TRUE,control=list(trace=1,REPORT=1))
SE  = sqrt(diag(solve(est$hessian)))
u = cbind(estimate=est$par, SE) 
rownames(u)=c('Phi11','sigw1','sigw2','sigv') 
u

Phi = diag(0,12)
Phi[1,1] = est$par[1]
Phi[2,]=c(0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1)
Phi[3,]=c(0,1,0,0,0,0,0,0,0,0,0,0)
Phi[4,]=c(0,0,1,0,0,0,0,0,0,0,0,0)
Phi[5,]=c(0,0,0,1,0,0,0,0,0,0,0,0)
Phi[6,]=c(0,0,0,0,1,0,0,0,0,0,0,0)
Phi[7,]=c(0,0,0,0,0,1,0,0,0,0,0,0)
Phi[8,]=c(0,0,0,0,0,0,1,0,0,0,0,0)
Phi[9,]=c(0,0,0,0,0,0,0,1,0,0,0,0)
Phi[10,]=c(0,0,0,0,0,0,0,0,1,0,0,0)
Phi[11,]=c(0,0,0,0,0,0,0,0,0,1,0,0)
Phi[12,]=c(0,0,0,0,0,0,0,0,0,0,1,0)
cQ1 = est$par[2]
cQ2 = est$par[3]
cQ  = diag(0,12)
cQ[1,1]=cQ1
cQ[2,2]=cQ2
cR  = est$par[4] 
ks = Ksmooth0(num, unemp, A, mu0, Sigma0, Phi, cQ, cR)

Tsm = ts(ks$xs[1,,], start=1948, freq=12)
Ssm = ts(ks$xs[2,,], start=1948, freq=12)
p1 = 3*sqrt(ks$Ps[1,1,])
p2 = 3*sqrt(ks$Ps[2,2,])
plot(Tsm, main='Trend Component', ylab='Trend')
xx = c(time(unemp), rev(time(unemp)))
yy = c(Tsm-p1, rev(Tsm+p1))
polygon(xx, yy, border=NA, col=gray(.5, alpha = .3))
plot(unemp, main='Data & Trend+Season', ylab='Unemp')
xx = c(time(unemp), rev(time(unemp)))
yy = c((Tsm+Ssm)-(p1+p2), rev((Tsm+Ssm)+(p1+p2)) )
polygon(xx, yy, border=NA, col=gray(.5, alpha = .3))

n.ahead = 36
y = ts(append(unemp, rep(0,n.ahead)), start=1948, freq=12)
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

par(mfrow=c(1,1))
plot(y, type='o', main='', ylab='unemp', xlim=c(1973,1982))
upp = ts(y[(num+1):(num+n.ahead)]+2*rmspe, start=1979, freq=12)
low = ts(y[(num+1):(num+n.ahead)]-2*rmspe, start=1979, freq=12)
xx = c(time(low), rev(time(upp)))
yy = c(low, rev(upp))
polygon(xx, yy, border=8, col=gray(.5, alpha = .3))
abline(v=1979, lty=3)



## Exercise P9

USINDPROD_M_NOV_2018 <- read_excel("USINDPROD_M_NOV_2018.xls", skip = 10)

df = ts(USINDPROD_M_NOV_2018[2], start = c(1919,1), end = c(2015,12),frequency = 12)

num = length(df)
A = cbind(1,1,0,0,0,0,0,0,0,0,0,0)

Linn =function(para){
  Phi = diag(0,12)
  Phi[1,1] = para[1]
  Phi[2,]=c(0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1)
  Phi[3,]=c(0,1,0,0,0,0,0,0,0,0,0,0)
  Phi[4,]=c(0,0,1,0,0,0,0,0,0,0,0,0)
  Phi[5,]=c(0,0,0,1,0,0,0,0,0,0,0,0)
  Phi[6,]=c(0,0,0,0,1,0,0,0,0,0,0,0)
  Phi[7,]=c(0,0,0,0,0,1,0,0,0,0,0,0)
  Phi[8,]=c(0,0,0,0,0,0,1,0,0,0,0,0)
  Phi[9,]=c(0,0,0,0,0,0,0,1,0,0,0,0)
  Phi[10,]=c(0,0,0,0,0,0,0,0,1,0,0,0)
  Phi[11,]=c(0,0,0,0,0,0,0,0,0,1,0,0)
  Phi[12,]=c(0,0,0,0,0,0,0,0,0,0,1,0)
  cQ1 = para[2]
  cQ2 = para[3]
  cQ  = diag(0,12)
  cQ[1,1]=cQ1
  cQ[2,2]=cQ2
  cR  = para[4]
  kf  = Kfilter0(num, df, A, mu0, Sigma0, Phi, cQ, cR)
  return(kf$like)
}

mu0 = rep(0,12); mu0[1] = df[1]
Sigma0 = diag(sd(df[1:12])/12,12)
init.par = c(1.03,.1,.1,.5)

est = optim(init.par, Linn,NULL, method='L-BFGS-B', hessian=TRUE,control=list(trace=1,REPORT=1))
SE  = sqrt(diag(solve(est$hessian)))
u = cbind(estimate=est$par, SE) 
rownames(u)=c('Phi11','sigw1','sigw2','sigv') 
u

Phi = diag(0,12)
Phi[1,1] = est$par[1]
Phi[2,]=c(0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1)
Phi[3,]=c(0,1,0,0,0,0,0,0,0,0,0,0)
Phi[4,]=c(0,0,1,0,0,0,0,0,0,0,0,0)
Phi[5,]=c(0,0,0,1,0,0,0,0,0,0,0,0)
Phi[6,]=c(0,0,0,0,1,0,0,0,0,0,0,0)
Phi[7,]=c(0,0,0,0,0,1,0,0,0,0,0,0)
Phi[8,]=c(0,0,0,0,0,0,1,0,0,0,0,0)
Phi[9,]=c(0,0,0,0,0,0,0,1,0,0,0,0)
Phi[10,]=c(0,0,0,0,0,0,0,0,1,0,0,0)
Phi[11,]=c(0,0,0,0,0,0,0,0,0,1,0,0)
Phi[12,]=c(0,0,0,0,0,0,0,0,0,0,1,0)
cQ1 = est$par[2]
cQ2 = est$par[3]
cQ  = diag(0,12)
cQ[1,1]=cQ1
cQ[2,2]=cQ2
cR  = est$par[4] 
ks = Ksmooth0(num, df, A, mu0, Sigma0, Phi, cQ, cR)

Tsm = ts(ks$xs[1,,], start=1919, freq=12)
Ssm = ts(ks$xs[2,,], start=1919, freq=12)
p1 = 3*sqrt(ks$Ps[1,1,])
p2 = 3*sqrt(ks$Ps[2,2,]) 
par(mfrow=c(1,1))
plot(Tsm, main='Trend Component', ylab='Trend')
xx = c(time(df), rev(time(df)))
yy = c(Tsm-p1, rev(Tsm+p1))
polygon(xx, yy, border=NA, col=gray(.5, alpha = .3))
plot(df, main='Data & Trend+Season', ylab='df')
xx = c(time(df), rev(time(df)))
yy = c((Tsm+Ssm)-(p1+p2), rev((Tsm+Ssm)+(p1+p2)) )
polygon(xx, yy, border=NA, col=gray(.5, alpha = .3))

n.ahead = 36
y = ts(append(df, rep(0,n.ahead)), start=1919, freq=12)
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

par(mfrow=c(1,1))
plot(y, type='o', main='', ylab='df', xlim=c(2010,2020),ylim=c(80,120))
upp = ts(y[(num+1):(num+n.ahead)]+2*rmspe, start=c(2016,1), freq=12)
low = ts(y[(num+1):(num+n.ahead)]-2*rmspe, start=c(2016,1), freq=12)
xx = c(time(low), rev(time(upp)))
yy = c(low, rev(upp))
polygon(xx, yy, border=8, col=gray(.5, alpha = .3))
abline(v=2016, lty=3)
df_temp = ts(tail(USINDPROD_M_NOV_2018[2],34), start = c(2016,1), end = c(2018,10),frequency = 12)
lines(df_temp, type='o',col='red')


# Exercise P10

plot(polio, type = 's') # view the data
acf2(polio)

model1 <- depmix(polio ~1, nstates=1, data=data.frame(polio),family=poisson())
model2 <- depmix(polio ~1, nstates=2, data=data.frame(polio),family=poisson())
model3 <- depmix(polio ~1, nstates=3, data=data.frame(polio),family=poisson())
model4 <- depmix(polio ~1, nstates=4, data=data.frame(polio),family=poisson())

fm1 <- fit(model1)
fm2 <- fit(model2)
fm3 <- fit(model3)
fm4 <- fit(model4)

plot(1:4,c(BIC(fm1),BIC(fm2),BIC(fm3),BIC(fm4)),ty="b")

summary(fm <- fit(model2))

u = as.vector(getpars(fm))
if (u[7] <= u[8]){
  para.mle = c(u[3:6], exp(u[7]), exp(u[8]))
}else{
  para.mle = c(u[6:3], exp(u[8]), exp(u[7]))
}
mtrans = matrix(para.mle[1:4], byrow=TRUE, nrow=2)
lams = para.mle[5:6]
pi1 = mtrans[2,1]/(2 - mtrans[1,1] - mtrans[2,2]);
pi2 = 1-pi1

plot(polio, main="", ylab="polio", type="h", col=gray(.7))
text(polio, col=6*posterior(fm)[,1]-2, labels=posterior(fm)[,1], cex=.9) 

plot(ts(posterior(fm)[,3], start=1900), ylab = expression(hat(pi)[~2]*'(t|n)')); abline(h=.5, lty=2)

hist(polio, breaks=30, prob=TRUE, main="")
xvals = seq(0,45)
u1 = pi1*dpois(xvals, lams[1])
u2 = pi2*dpois(xvals, lams[2])
lines(xvals, u1, col=4)
lines(xvals, u2, col=2)

pois.HMM.generate_sample = function(n,m,lambda,Mtrans,StatDist=NULL){
  if(is.null(StatDist)) StatDist = solve(t(diag(m)-Mtrans +1),rep(1,m))
  mvect = 1:m
  state = numeric(n)
  state[1] = sample(mvect ,1, prob=StatDist)
  for (i in 2:n)
    state[i] = sample(mvect ,1,prob=Mtrans[state[i-1] ,])
  y = rpois(n,lambda=lambda[state ])
  list(y= y, state= state)
}

nboot = 100
nobs = length(polio)
para.star = matrix(NA, nrow=nboot, ncol = 6)
for (j in 1:nboot){
  x.star = pois.HMM.generate_sample(n=nobs, m=2, lambda=lams, Mtrans=mtrans)$y
  model2 <- depmix(x.star ~1, nstates=2, data=data.frame(x.star),
                  family=poisson())
  u = as.vector(getpars(fit(model2, verbose=0)))
  if (u[7] <= u[8]) { para.star[j,] = c(u[3:6], exp(u[7]), exp(u[8])) }
  else  { para.star[j,] = c(u[6:3], exp(u[8]), exp(u[7])) }     
}

SE = sqrt(apply(para.star,2,var) +
            (apply(para.star,2,mean)-para.mle)^2)[c(1,4:6)]
names(SE)=c('seM11/M12', 'seM21/M22', 'seLam1', 'seLam2')
SE