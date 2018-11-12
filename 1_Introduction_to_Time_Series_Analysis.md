# Introduction to Time Series Analysis

## Real Life Examples

The first step in any time series investigation always involves careful scrutiny of the recorded data plotted over time

### Johnson & Johnson

![Johnson & Johnson](./Images/johnson_johnson.png)

This picture shows quarterly earnings per share for the U.S. company Johnson&Johnson.

In this case we want to focus our attention on the __increasing underlying trend variability__, and somewhat __regular oscillation__ superimposed on the trend that seems to repeat over quarters.

```R
library(astsa)
tsplot(jj, type="o", ylab="Quarterly  Earnings  per  Share")
tsplot(log(jj)) # not  shown
```

### S&P100 Index

![S&P100 Index](./Images/S&P100.png)

This picture shows prices and daily returns of the Standard and Poor’s 100 Index (S&P100) from 1984 to 2017

It is easy to spot the financial crisis of 2008 in the figure

The mean of the series appears to be stable with an average return of apporximately zero, however, the __volatility__ (or __variability__) of data __exhibits clustering__; that is __highly volative periods tend to be clustered together__

```R
library(xts)
djiar = diff(log(djia$Close))[-1] # approximate  returns
tsplot(djiar , main="DJIA  Returns", xlab=’’, margins=.5)
```

### USD/GBP Foreign exchange rate

![USD/GBP Foreign exchange rate](./Images/USD_GBP.png)

This image shows the weekly USD/GBP foreign exchange rate (U.S. Dollars to One British Pound)

```R
library("fImport")
usbp=fredSeries("DEXUSEU", from="2001-01-01")
tsplot(usbp)
```

### Crypto currencies

![Crypto currencies](./Images/crypto_currencies.png)

This image shows Cryptocurrency “BitCoin” from April 28, 2013 to November 25, 2017

## Time series methods

The primary objective of time series analysis is to develop mathematical models that provide plausible descriptions for sample data

### White noise

A simple kind of generated series might be a collection of uncorrelated random variables, _wt_, with __mean 0__ and __finite variance σ^2__. We denote this process as __ε_t ~ N(0,σ^2)__

The time series generated from uncorrelated variables is used as a model for noise in engineering applications where it is called white noise

We often require stronger conditions and need the noise to be __Gaussian white noise__, where in the ε_t are independent and identically distributed (iid) normal random variables, with __mean 0__ and __variance σ^2__

Although both cases require __0 mean__ and __constant variance__, the difference is that generically, the term white noise means the time series is uncorrelated. __Gaussian white noise__ implies __normality__ (which implies __independence__)

If the stochastic behaviour of all time series could be explained in terms of the __white noise model__, classical statistical methods would suffice.

### Moving averages and filters

We might __replace__ the __white noise series wt__ __by__ a __moving average__ that __smooths the series__

e.g.

![moving averages and filters](./Images/moving_average_filters.png)

```R
w = rnorm (500,0,1) # 500 N(0,1)  variates
v = filter(w, sides=2, rep (1/3 ,3)) # moving  average
par(mfrow=c(2,1))
tsplot(w, main="white  noise")
tsplot(v, ylim=c(-3,3), main="moving  average")
```

This series is much smoother than the white noise series, and it is apparent that averaging removes some of the high frequency behaviour of the noise.

A linear combination of values in a time series such as the above equation is referred to, generically, a filtered series; hence the command filter.

### Autoregressions

Suppose we consider the __white noise series ε_t__ as input and calculate the output using the __second order equation__

![Autoregressions second order equation](./Images/autoregression_second_order_equation.png)

Successively for t = 1,2... the previous equation represents a __regression or prediction__ fo the current value __x_t__ of a time series as a __function of the past two values__ of the series, and, hence, the term autoregression is suggested for this model.

A problem with startup values exists here because the equation also depends on the initial conditions x_0 and x_-1, but for now assume they are 0. We can then generate data recursively by substituting into the previuos formula

```R
w = rnorm (550,0,1) # 50  extra  to  avoid  startup  problems
x = filter(w, filter=c(1,-.9), method="recursive")[-(1:50)]
tsplot(x, main="autoregression")
```

### Random walk with drift

![random walk with drift](./Images/rand_walk_drift.png)

for t = 1,2,... with initial condition x_0 = 0, ans where ε_t is white noise. The constant δ is called the drift, and when δ = 0, the model is called simply a random walk because the value of the time t is the value of the series at time t-1 plus a completely random movement determined by ε_t. Note that we may rewrite the previous formula as a cumulative sum of white noise variates.

![rand_walk_drift_sum_white_noise](./Images/rand_walk_drift_sum_white_noise.png)

for t = 1,2,...

```R
set.seed(154) # so you can reproduce the results
w = rnorm(200); x = cumsum(w) # two commands in one line
wd = w + .2; xd = cumsum(wd)
ts.plot(xd, ylim = c(-5, 55), main = "random walk", ylab ='')
abline(a = 0, b= .2, lty = 2) # drift
lines (x, col = 4)
abline (h = 0, col = 4, lty = 2)
```

### Measures of Dependence

We now discuss various measures that describe the general behavior of a process as it evolves over time. A rather simple descriptive measure is the mean function, such as the average monthly high temperature for your city. In this case, the mean is a function of time.

#### The mean function

![mean function](./Images/mean_func.png)

#### The autocovariance function

![autocovariance function](./Images/autocov_function.png)

The autocovariance measures the linear dependence between two points on the same series observed at different times

If the result is 0, then x_s and x_t are not linearly related, but there may be some dependence structure between them.

![autocov_function_same_t](./Images/autocov_function_same_t.png)

#### The autocorrelation function (ACF)

![autocorrelation function](./Images/autocorrelation_func.png)

The ACF measures the linear predictability of the series at time _t_, say x_ _t_, using only the value x_ _s_.

The ACF has values in [-1,1], easily shown with Cauchy-Schwarz inequality

If we can predict x_ _t_ perfectly from x_ _s_ through a linear relationship, x_ _t_ = β_0 + β_1 * x_ _s_, then the correlation will be +1 when β_1 > 0, and -1 when β_1 < 0

### Stationary time series

