library(rstan)
library(ggplot2)
source("nes.data.R", echo = TRUE)    

 # Estimation (nes.stan)
 # glm (vote ~ income, family=binomial(link="logit"))
if (!exists("nes.sm")) {
    if (file.exists("nes.sm.RData")) {
        load("nes.sm.RData", verbose = TRUE)
    } else {
        rt <- stanc("nes.stan", model_name = "nes")
        nes.sm <- stan_model(stanc_ret = rt)
        save(nes.sm, file = "nes.sm.RData")
    }
}

dataList.1 <- c("N","income","vote")
nes.sf1 <- sampling(nes.sm, dataList.1)
print(nes.sf1)

beta.post <- extract(nes.sf1, "beta")$beta
beta.mean <- colMeans(beta.post)

## Evaluation at the mean

invlogit(-1.40 + 0.33*mean(income, na.rm=T))

## Fitting and displaying the model

 # see file "5.1_Logistic regression with one predictor.R" for the commands
 # to plot figure 5.1 (a) & (b)

## Displaying the results of several logistic regressions

#clean data for graph
brdata <- read.dta("nes5200_processed_voters_realideo.dta",convert.factors=F)
# Data are at http://www.stat.columbia.edu/~gelman/arm/examples/nes

 # Clean the data
brdata <- brdata[is.na(brdata$black)==FALSE&is.na(brdata$female)==FALSE&is.na(brdata$educ1)==FALSE
&is.na(brdata$age)==FALSE&is.na(brdata$income)==FALSE&is.na(brdata$state)==FALSE,]
kept.cases <- 1952:2000
matched.cases <- match(brdata$year, kept.cases)
keep <- !is.na(matched.cases)
data <- brdata[keep,]
plotyear <- unique(sort(data$year))
year.new <- match(data$year,unique(data$year))
n.year <- length(unique(data$year))
income.new <-data$income-3
age.new <- (data$age-mean(data$age))/10
y <- data$rep_pres_intent
data <- cbind(data, year.new, income.new, age.new, y)
nes.year <- data[,"year"]
age.discrete <- as.numeric (cut (data[,"age"], c(0,29.5, 44.5, 64.5, 200)))
race.adj <- ifelse (data[,"race"]>=3, 1.5, data[,"race"])
data <- cbind (data, age.discrete, race.adj)

female <- data[,"gender"] - 1
black <- ifelse (data[,"race"]==2, 1, 0)
rvote <- ifelse (data[,"presvote"]==1, 0, ifelse(data[,"presvote"]==2, 1, NA))

region.codes <- c(3,4,4,3,4,4,1,1,5,3,3,4,4,2,2,2,2,3,3,1,1,1,2,2,3,2,4,2,4,1,1,4,1,3,2,2,3,4,1,
   1,3,2,3,3,4,1,3,4,1,2,4)
attach.all(data)

income.year <- NULL
income.coef <- NULL
income.se <- NULL
for (yr in seq(1952,2000,4)){
  ok <- nes.year==yr & presvote<3
  vote <- data$presvote[ok] - 1
  income <- data$income[ok]
  fit.1 <- glm (vote ~ income, family=binomial(link="logit"))
  income.year <- c (income.year, yr)
  income.coef <- c (income.coef, fit.1$coef[2])
  income.se <- c (income.se, se.coef(fit.1)[2])
}

 # Figure 5.4
y.max<-NULL
y.min<-NULL
for (i in 1:13) {
  y.bounds <- income.coef[i]+income.se[i]*c(-1,1)
  y.max <-c(y.max,y.bounds[2])
  y.min <-c(y.min,y.bounds[1])
}

limits <- aes(ymax=y.max,ymin=y.min)
frame1 = data.frame(year=income.year,coeff=income.coef)
p1 <- ggplot(frame1,aes(x=year,y=coeff)) +
      geom_point() +
      scale_y_continuous("Coefficient of Income",limits=c(-.05,.52)) +
      scale_x_continuous("Year",limits=c(1950,2000)) +
      theme_bw() +
      geom_pointrange(limits) +
      geom_hline(yintercept=0,linetype="dashed")
print(p1)
