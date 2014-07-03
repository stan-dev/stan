## Read the data
## Read the the Social Indicators Survey data 
# Data are at http://www.stat.columbia.edu/~gelman/arm/examples/sis

# The R codes & data files should be saved in the same directory for
# the source command to work

source("25.4_RadomImputationofaSingleVariable.R") # where variables were defined

## Variables description:

 # interest:  interest of entire family
interest <- na.fix(interest)

 # transforming the different sources of income
interest <- interest/1000

## Simple random imputation
interest.imp <- random.imp (interest)

## Iterative regression imputation
impute <- function (a, a.impute){
   ifelse (is.na(a), a.impute, a)
}

n.sims <- 10
for (s in 1:n.sims) {
 earnings1 <- random.imp(earnings)
 earnings.top1 <- random.imp(earnings.top)
 interest.imp1 <- random.imp(interest.imp)
 male1 <- random.imp(male)
 over651 <- random.imp(over65)
 white1 <- random.imp(white)
 immig1 <- random.imp(immig)
 educ_r1 <- random.imp(educ_r)
 workmos1 <- random.imp(workmos)
 workhrs.top1 <- random.imp(workhrs.top)
 any.ssi1 <- random.imp(any.ssi)
 any.welfare1 <- random.imp(any.welfare)
 any.charity1 <- random.imp(any.charity)
 sis <- list(N=length(earnings1),earnings=earnings1,interest=interest.imp1,
             earnings_top=earnings.top1, male=male1,
             over65=over651,white=white1, immig=immig1, educ_r=educ_r1,
             workmos=workmos1, workhrs_top=workhrs.top1,
             any_ssi=any.ssi1, any_welfare=any.welfare1,
             any_charity=any.charity1)

 earnings2.sf1 <- stan(file='earnings2.stan', data=sis, iter=1000, chains=4)
 print(earnings2.sf1)
 post <- extract(earnings.sf1)
 beta <- colMeans(post$beta)

 pred.1 <- beta[1] + beta[2] * interest.imp1 + beta[3] * male1 + beta[4] * over651 + beta[5] * white1 + beta[6] * immig1 + beta[7] * educ_r1 + beta[8] * workmos1 + beta[9] * workhrs.top1 + beta[10] * any.ssi1 + beta[11] * any.welfare1 + beta[12] * any.charity1

 pred.1 <- rnorm (n, pred.1, sd(pred.1))
 earnings.imp <- impute (earnings, pred.1)

 earnings.imp1 <- random.imp(earnings.imp)
 sis <- list(N=length(earnings1),earnings=earnings1,interest=earnings.imp1,
             earnings_top=earnings.top1, male=male1,
             over65=over651,white=white1, immig=immig1, educ_r=educ_r1,
             workmos=workmos1, workhrs_top=workhrs.top1,
             any_ssi=any.ssi1, any_welfare=any.welfare1,
             any_charity=any.charity1)
 
 earnings2.sf1 <- stan(file='earnings2.stan', data=sis, iter=1000, chains=4)
 print(earnings2.sf1)
 post <- extract(earnings.sf1)
 beta <- colMeans(post$beta)

 pred.2 <- beta[1] + beta[2] * earnings.imp1 + beta[3] * male1 + beta[4] * over651 + beta[5] * white1 + beta[6] * immig1 + beta[7] * educ_r1 + beta[8] * workmos1 + beta[9] * workhrs.top1 + beta[10] * any.ssi1 + beta[11] * any.welfare1 + beta[12] * any.charity1
 
  pred.2 <- rnorm (n, pred.2, sd(pred.2))
  interest.imp <- impute (interest, pred.2)
}



