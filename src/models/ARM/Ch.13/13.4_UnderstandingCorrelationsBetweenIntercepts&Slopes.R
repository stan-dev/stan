library(rstan)
library(ggplot2)
library(foreign)
## Read the data FIXME: run models and test graphics
# Data are at http://www.stat.columbia.edu/~gelman/arm/examples/radon

# The R codes & data files should be saved in the same directory for
# the source command to work

heights <- read.dta ("heights.dta")
attach(heights)

  # define variables 
age <- 90 - yearbn                     # survey was conducted in 1990
age[age<18] <- NA
age.category <- ifelse (age<35, 1, ifelse (age<50, 2, 3))
eth <- ifelse (race==2, 1, ifelse (hisp==1, 2, ifelse (race==1, 3, 4)))
male <- 2 - sex

  # (for simplicity) remove cases with missing data
ok <- !is.na (earn+height+sex) & earn>0 & yearbn>25
heights.clean <- as.data.frame (cbind (earn, height, sex, race, hisp, ed, age,
    age.category, eth, male)[ok,])
n <- nrow (heights.clean)
attach.all (heights.clean)
height.jitter.add <- runif (n, -.2, .2)

 # rename variables
y <- log(earn)
x <- height
n <- length(y)
n.age <- 3
n.eth <- 4
age <- age.category

## Regression of log (earnings) on height, age, and ethnicity
## M1 <- lmer (y ~ x + (1 + x | eth))
if (!exists("earnings_vary_si.sm")) {
    if (file.exists("earnings_vary_si.sm.RData")) {
        load("earnings_vary_si.sm.RData", verbose = TRUE)
    } else {
        rt <- stanc("earnings_vary_si.stan", model_name = "earnings_vary_si")
        earnings_vary_si.sm <- stan_model(stanc_ret = rt)
        save(earnings_vary_si.sm, file = "earnings_vary_si.sm.RData")
    }
}

dataList.1 <- list(N=n, earn=earn,height=x,ethn=ethn)
earnings_vary_si.sf1 <- sampling(earnings_vary_si.sm, dataList.1)
print(earnings_vary_si.sf1)
post1 <- extract(earnings_vary_si.sf1)
post1.ranef <- colMeans(post1$const_coef)
mean.ranef1 <- mean(post1.ranef)
post1.beta <- colMeans(post1$beta)
post1.ranef2 <- mean(post1.beta)
mean.ranef2 <- mean(post1.ranef2)

##FIXME: WHAT'S THIS
ab.hat.M1 <- coef(M1)$eth
a.hat.M1 <- ab.hat.M1[,1]
b.hat.M1 <- ab.hat.M1[,2]

## plot on figure 13.3
x.jitter <- x + runif(n, -.2,.2)
age.label <- c("age 18-34", "age 35-49", "age 50-64")
eth.label <- c("blacks", "hispanics", "whites", "others")
x <- height - mean(height)
x.jitter.add <- runif(n, -.2,.2)
display8 <- c (1,2,3,4)  # ethnicities to be displayed
y.range <- range (y[!is.na(match(county,display8))])

eth.data <- data.frame(y, x, eth)
eth.data$x <- x + x.jitter.add
eth.data$eth.name <- eth.data$eth
eth.data$eth.name <- factor(radon8.data$eth.name,levels=c("1","2","3","4"),
                                  labels=c("BLACKS","HISPANICS","WHITES","OTHERS"))
eth.data$a <- 
eth.data$b <-
eth.data$a.hat <-
eth.data$b.hat <-

p1 <- ggplot(radon8.data, aes(x, y)) +
     geom_jitter(position = position_jitter(width = .05, height = 0)) +
     scale_x_continuous("Height (inches)") +
     scale_y_continuous("log(earnings)") +
     geom_abline(aes(intercept = a.hat, slope=b.hat), size = 0.25) +
for (s in 1:20)
  p1 <- p1 + geom_abline(intercept=a[s,], slope=b[s,])
p1 <- p1 + facet_wrap(~ eth.name, ncol = 4)
print(p1)

## plot on figure 13.4
dev.new()
frame2 = data.frame(x1=a.hat.M1,y1=b.hat.M1)
p2 <- ggplot(frame2,aes(x=x1,y=y1)) +
      geom_point() +
      scale_y_continuous("Intercept") +
      scale_x_continuous("Slope") +
      theme_bw() +
print(p2)

 # plot on figure 13.5
dev.new()
frame3 = data.frame(x1=x.jitter,y1=y)
p3 <- ggplot(frame3,aes(x=x1,y=y1)) +
      geom_point() +
      scale_y_continuous("log(earnings)") +
      scale_x_continuous("Height (inches)") +
      geom_abline(intercept=a.hat.M1[1],slope=b.hat.M1[1]) +
      geom_abline(intercept=a.hat.M1[2],slope=b.hat.M1[2]) +
      geom_abline(intercept=a.hat.M1[3],slope=b.hat.M1[3]) +
      geom_abline(intercept=a.hat.M1[4],slope=b.hat.M1[4]) +
      theme_bw() +
print(p3)

## Regression centering the predictors
## M2 <- lmer (y ~ x.centered + (1 + x.centered | eth))
x.centered <- x - mean(x)
x.centered.jitter <- x.jitter - mean(x)

dataList.3 <- list(N=n, earn=earn,height=x.centered,ethn=ethn)
earnings_vary_si.sf2 <- sampling(earnings_vary_si.sm, dataList.3)
print(earnings_vary_si.sf2)
post1 <- extract(earnings_vary_si.sf2)
post1.ranef <- colMeans(post1$const_coef)
mean.ranef1 <- mean(post1.ranef)
post1.beta <- colMeans(post1$beta)
post1.ranef2 <- mean(post1.beta)
mean.ranef2 <- mean(post1.ranef2)

dev.new()
eth2.data <- data.frame(y, x, eth)
eth2.data$x <- x + x.jitter.add
eth2.data$eth.name <- eth.data$eth
eth2.data$eth.name <- factor(radon8.data$eth.name,levels=c("1","2","3","4"),
                                  labels=c("BLACKS","HISPANICS","WHITES","OTHERS"))
eth2.data$a <- 
eth2.data$b <-
eth2.data$a.hat.M2 <-
eth2.data$b.hat.M2 <-

p4 <- ggplot(eth2.data, aes(x, y)) +
     geom_jitter(position = position_jitter(width = .05, height = 0)) +
     scale_x_continuous("Height (inches from mean)") +
     scale_y_continuous("log(earnings)") +
     geom_abline(aes(intercept = a.hat.M2, slope=b.hat.M2), size = 0.25) +
for (s in 1:20)
  p4 <- p4 + geom_abline(intercept=a[s,], slope=b[s,])
p4 <- p4 + facet_wrap(~ eth.name, ncol = 4)
print(p4)

## Figure 13.7
dev.new()
frame5 = data.frame(x1=a.hat.M2,y1=b.hat.M2)
p5 <- ggplot(frame5,aes(x=x1,y=y1)) +
      geom_point() +
      scale_y_continuous("Intercept") +
      scale_x_continuous("Slope") +
      theme_bw() +
print(p5)
