## Read the the Social Indicators Survey data 
# Data are at http://www.stat.columbia.edu/~gelman/arm/examples/sis
library(rstan)
library(ggplot2)

wave3 <- read.table ("siswave3v4impute3.csv", header=T, sep=",")
attach(wave3)
n <- nrow (wave3)

 # missing codes:  -9: refused/dk to say if you have this source
 #                 -5: you said you had it but refused/dk the amount

 # Variables description:

 # rearn:  respondent's earnings
 # tearn:  spouse's earnings 
 # ssi:  ssi for entire family
 # welfare:  public assistance for entire family
 # charity:  income received from charity for entire family
 # sex:  male=1, female=2
 # race of respondent:  1=white, 2=black, 3=hispanic(nonblack), 4=other
 # immig:  0 if respondent is U.S. citizen, 1 if not
 # educ_r:  respondent's education (1=no hs, 2=hs, 3=some coll, 4=college grad)
 # DON'T USE primary:  -9=missing, 0=spouse, 1=respondent is primary earner  
 # workmos:  primary earner's months worked last year
 # workhrs:  primary earner's hours/week worked last year


white <- ifelse (race==1, 1, 0)
white[is.na(race)] <- 0
male <- ifelse (sex==1, 1, 0)
over65 <- ifelse (r_age>65, 1, 0)
immig[is.na(immig)] <- 0
educ_r[is.na(educ_r)] <- 2.5

 # set up some simplified variables to work with
na.fix <- function (a) {
  ifelse (a<0 | a==999999, NA, a)
}

is.any <- function (a) {
  any.a <- ifelse (a>0, 1, 0)
  any.a[is.na(a)] <- 0
  return(any.a)
}

workmos <- workmos
earnings <- na.fix(rearn) + na.fix(tearn)
earnings[workmos==0] <- 0

 # summary variables for various income supports
any.ssi <- is.any (ssi)
any.welfare <- is.any (welfare)
any.charity <- is.any (charity)


 # transforming the different sources of income
earnings <- earnings/1000

## Simple random imputation
random.imp <- function (a){
  missing <- is.na(a)
  n.missing <- sum(missing)
  a.obs <- a[!missing]
  imputed <- a
  imputed[missing] <- sample (a.obs, n.missing, replace=TRUE)
  return (imputed)
}

earnings.imp <- random.imp (earnings)

## Zero coding and topcoding
topcode <- function (a, top){
  return (ifelse (a>top, top, a))
}

earnings.top <- topcode (earnings, 100)
workhrs.top <- topcode (workhrs, 40)

 # plot figure 25.1a
frame = data.frame(earnings = earnings.top[earnings>0])
  p1 <- ggplot(frame,aes(earnings)) +
        geom_histogram(colour = "black", fill = "white", binwidth=5) +
        theme_bw() +
        labs(title="Observed earnings (excluding 0's)") 
print(p1)

## Using regression predictions to perform deterministic imputation

 # create a little dataset with all redefined variables
earnings1 <- random.imp(earnings)
earnings.top1 <- random.imp(earnings.top)
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
sis <- list(N=length(earnings1),earnings=earnings1,
            earnings_top=earnings.top1, male=male1,
            over65=over651,white=white1, immig=immig1, educ_r=educ_r1,
            workmos=workmos1, workhrs_top=workhrs.top1,
            any_ssi=any.ssi1, any_welfare=any.welfare1,
            any_charity=any.charity1)

earnings.sf1 <- stan(file='earnings.stan', data=sis, iter=1000, chains=4)
print(earnings.sf1)
post <- extract(earnings.sf1)
beta <- colMeans(post$beta)

 # get predictions
pred.1 <- beta[1] + beta[2] * male1 + beta[3] * over651 + beta[4] * white1 + beta[5] * immig1 + beta[6] * educ_r1 + beta[7] * workmos1 + beta[8] * workhrs.top1 + beta[9] * any.ssi1 + beta[10] * any.welfare1 + beta[11] * any.charity1

 # function to create a completed dataset by imputing the predictions
impute <- function (a, a.impute){
   ifelse (is.na(a), a.impute, a)
}
earnings.imp.1 <- impute (earnings, pred.1)  # use it to inpute missing
                                             # earnings

## Transforming and topcoding
earnings.sqrt <- I(sqrt(earnings.top))
earnings.sqrt <- random.imp(earnings.sqrt)

sis2 <- list(N=length(earnings.sqrt),earnings=earnings.sqrt,
            earnings_top=earnings.top1, male=male1,
            over65=over651,white=white1, immig=immig1, educ_r=educ_r1,
            workmos=workmos1, workhrs_top=workhrs.top1,
            any_ssi=any.ssi1, any_welfare=any.welfare1,
            any_charity=any.charity1)

earnings.sf2 <- stan(file='earnings.stan', data=sis2, iter=1000, chains=4)
print(earnings.sf2)

post2 <- extract(earnings.sf2)
beta2 <- post2$beta
pred.2.sqrt <- beta2[1] + beta2[2] * male1 + beta2[3] * over651 + beta2[4] * white1 + beta2[5] * immig1 + beta2[6] * educ_r1 + beta2[7] * workmos1 + beta2[8] * workhrs.top1 + beta2[9] * any.ssi1 + beta2[10] * any.welfare1 + beta2[11] * any.charity1
pred.2 <- topcode (pred.2.sqrt^2, 100)
earnings.imp.2 <- impute (earnings.top, pred.2)

 # plot figure 25.1b
dev.new()
frame2 = data.frame(earnings = earnings.imp.2[is.na(earnings)])
  p2 <- ggplot(frame2,aes(earnings)) +
        geom_histogram(colour = "black", fill = "white",binwidth=7) +
        theme_bw() +
        labs(title="Deterministic imputation of earnings") 
print(p2)

## Random regression imputation
pred.4.sqrt <- rnorm (n, pred.2.sqrt, sd(pred.2.sqrt))
pred.4 <- topcode (pred.4.sqrt^2, 100)
earnings.imp.4 <- impute (earnings.top, pred.4)

 # plot figure 25.1c
dev.new()
frame3 = data.frame(earnings = earnings.imp.4[is.na(earnings)])
  p3 <- ggplot(frame3,aes(earnings)) +
        geom_histogram(colour = "black", fill = "white", binwidth=5) +
        theme_bw() +
        labs(title="Random imputation of earnings") 
print(p3)

## Plots figure 25.2
frame4 = data.frame(x1=earnings.imp.2[is.na(earnings)],y2=earnings.imp.4[is.na(earnings)],y1=earnings.imp.4[is.na(earnings)])
frame5 = data.frame(x2=pred.2[!is.na(earnings)],y2=earnings[!is.na(earnings)])
dev.new()
p4 <- ggplot() +
      geom_jitter(position=position_jitter(height=.08,width=.4)) +
      geom_point(data=frame4,aes(x=x1,y=y1)) +
      geom_point(data=frame5,aes(x=x2,y=y2),colour="grey30") +
      labs(title='Random Imputation') +
      scale_x_continuous("Regression Prediction") +
      scale_y_continuous("Imputed Income")
print(p4)

dev.new()
p5 <- ggplot() +
      geom_jitter(position=position_jitter(height=.08,width=.4)) +
      geom_point(data=frame4,aes(x=x1,y=y2)) +
      geom_point(data=frame5,aes(x=x2,y=y2),colour="grey30") +
      labs(title='Random Imputation') +
      scale_x_continuous("Regression Prediction") +
      scale_y_continuous("Imputed Income")
print(p5)

## Two-stage imputation model

 # fit the 2 models
earnings1 <- I(earnings>0);
earnings1 <- ifelse(earnings1==1,1,0)
ok <- !is.na(earnings1+earnings.top1+male1+over651+white1+immig1+educ_r1+any.ssi1+any.welfare1+any.charity1)
sis <- list(N=length(earnings1[ok]),earnings=earnings1[ok],
            earnings_top=earnings.top1[ok], male=male1[ok],
            over65=over651[ok],white=white1[ok], immig=immig1[ok],
            educ_r=educ_r1[ok],
            any_ssi=any.ssi1[ok], any_welfare=any.welfare1[ok],
            any_charity=any.charity1[ok])


earnings_pt1.sf1 <- stan(file='earnings_pt1.stan', data=sis, iter=1000,
                         chains=4)
print(earnings_pt1.sf1)
post <- extract(earnings_pt1.sf1)
beta <- colMeans(post$beta)

earnings1 <- I(sqrt(earnings.top));
ok <- !is.na(earnings1+earnings.top1+male1+over651+white1+immig1+educ_r1+any.ssi1+any.welfare1+any.charity1)
sis <- list(N=length(earnings1[ok]),earnings=earnings1[ok],
            earnings_top=earnings.top1[ok], male=male1[ok],
            over65=over651[ok],white=white1[ok], immig=immig1[ok],
            educ_r=educ_r1[ok],
            any_ssi=any.ssi1[ok], any_welfare=any.welfare1[ok],
            any_charity=any.charity1[ok])

earnings_pt2.sf1 <- stan(file='earnings_pt2.stan', data=sis, iter=1000,
                         chains=4)
print(earnings_pt2.sf1)
post2 <- extract(earnings_pt2.sf1)
beta2 <- colMeans(post2$beta)

pred.glm <- beta[1] + beta[2] * male + beta[3] * over65 + beta[4] * white + beta[5] * immig + beta[6] * educ_r + beta[7] * any.ssi + beta[8] * any.welfare + beta[9] * any.charity

pred.lm <- beta2[1] + beta2[2] * male + beta2[3] * over65 + beta2[4] * white + beta2[5] * immig + beta2[6] * educ_r + beta2[7] * any.ssi + beta2[8] * any.welfare + beta2[9] * any.charity

pred.sign <- rbinom (n, 1, pred.glm)
pred.pos.sqrt <- rnorm (n,pred.lm,sd(pred.lm))
pred.pos <- topcode (pred.pos.sqrt^2, 100)
earnings.imp <- impute (earnings, pred.sign*pred.pos)
