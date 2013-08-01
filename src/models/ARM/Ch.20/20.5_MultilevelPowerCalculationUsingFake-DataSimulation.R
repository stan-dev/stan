library(rstan)
library(ggplot2)

hiv.data <- read.csv ("allvar.csv")
attach(hiv.data)

## Eliminate missing data & just consider the "control" patients (treatmnt==1)
## and with initial age between 1 and 5 years
ok <- treatmnt==1 & !is.na(CD4PCT) & (baseage>1 & baseage<5)
attach(hiv.data[ok,])

## Redefining variables
y <- sqrt (CD4PCT)
age.baseline <- baseage        # kid's age (yrs) at the beginning of the study
age.measurement <- visage      # kids age (yrs) at the time of measurement
treatment <- treatmnt
time <- visage - baseage

## Set up new patient id numbers from 1 to J
ok <- !is.na(y+time+person)

unique.pid <- unique (newpid)
n <- length (y[ok])
J <- length (unique.pid[ok])
person <- rep (NA, n)
for (j in 1:J){
person[newpid==unique.pid[j]] <- j
}

## Fit the model
## M1 <- lmer (y ~ time + (1 + time | person))
if (!exists("hiv.sm")) {
    if (file.exists("hiv.sm.RData")) {
        load("hiv.sm.RData", verbose = TRUE)
    } else {
        rt <- stanc("hiv.stan", model_name = "hiv")
        hiv.sm <- stan_model(stanc_ret = rt)
        save(hiv.sm, file = "hiv.sm.RData")
    }
}

dataList.1 <- list(N=n, J=84,time=time[ok],person=person[ok],y=y[ok])
hiv.sf1 <- sampling(hiv.sm, dataList.1)
print(hiv.sf1,pars = c("a","b", "sigma_y", "lp__"))
post <- extract(hiv.sf1)

## Simulating the hypothetical data
CD4.fake <- function(J, K){
  time <- rep (seq(0,1,length=K), J)  # K measurements during the year
  person <- rep (1:J, each=K)         # person ID's
  treatment <- sample (rep(0:1, J/2))
  treatment1 <- treatment[person] 
#                                     # hyperparameters
  mu.a.true <- 4.8                    # more generally, these could
  g.0.true <- -.5                     # be specified as additional
  g.1.true <- .5                      # arguments to the function
  sigma.y.true <- .7
  sigma.a.true <- 1.3
  sigma.b.true <- .7
#                                     # personal-level parameters
  a.true <- rnorm (J, mu.a.true, sigma.a.true)
  b.true <- rnorm (J, g.0.true + g.1.true*treatment, sigma.b.true)
#                                     # data
  y <- rnorm (J*K, a.true[person] + b.true[person]*time, sigma.y.true)
  return (data.frame (y=y, time=time, person=person, treatment=treatment1,J=84,N=length(y)))
}
## Fitting the model and checking the power
if (!exists("hiv_inter.sm")) {
    if (file.exists("hiv_inter.sm.RData")) {
        load("hiv_inter.sm.RData", verbose = TRUE)
    } else {
        rt <- stanc("hiv_inter.stan", model_name = "hiv_inter")
        hiv_inter.sm <- stan_model(stanc_ret = rt)
        save(hiv_inter.sm, file = "hiv_inter.sm.RData")
    }
}

CD4.power <- function (J, K, n.sims=1000){
  signif <- rep (NA, n.sims)
  for (s in 1:n.sims){
    fake <- CD4.fake (J,K)
    hiv.sf2 <- sampling(hiv_inter.sm, fake)
    post <- extract(hiv.sf2)
    theta.hat <- colMeans(post$beta)
    theta.se <- sd(post$beta)
    signif[s] <- (theta.hat - 2*theta.se) > 0    # return TRUE or FALSE
  }
  power <- mean (signif)                         # proportion of TRUE
  return (power)
}

## Figure 20.5 (a)
frame = data.frame(x1=time,y1=y,newpid=newpid)
p1 <- ggplot(frame,aes(x=x1,y=y1)) +
    geom_point() +
    scale_y_continuous("sqrt (CD4%)") +
    scale_x_continuous("Time (years)") +
    theme_bw() +
    labs(title="Observed Data")
for (j in 1:84) {
  p1 <- p1 + geom_line(data=frame[frame$newpid==unique.pid[j],])
}
print(p1)

## Figure 20.5 (b)
coef.1 <- colMeans(post$a)
coef.2 <- colMeans(post$b)
dev.new()

frame = data.frame(x1=time,y1=y,newpid=newpid)
p2 <- ggplot(frame,aes(x=x1,y=y1)) +
    geom_point() +
    scale_y_continuous("sqrt (CD4%)") +
    scale_x_continuous("Time (years)") +
    theme_bw() +
    labs(title="Estimated Trend Lines")
for (j in 1:84) {
  p2 <- p2 + geom_abline(intercept=coef.1[j],slope=coef.2[j])
}
print(p2)
