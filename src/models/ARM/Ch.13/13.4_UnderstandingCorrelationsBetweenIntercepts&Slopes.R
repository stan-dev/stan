library(rstan)
library(ggplot2)
library(foreign)
library(gridBase)
## Read the data 
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
height.jitter.add <- runif (n, -.2, .2)

 # rename variables
y <- log(earn[ok])
x <- height[ok]
n <- length(y[ok])
n.age <- 3
n.eth <- 4
age <- age.category
eth.ok <- eth[ok]

## Regression of log (earnings) on height, age, and ethnicity
## M1 <- lmer (y ~ x + (1 + x | eth.ok))
dataList.1 <- list(N=n, earn=earn[ok]+1,height=x,eth=eth[ok])
earnings_vary_si.sf1 <- stan(file='earnings_vary_si.stan', data=dataList.1,
                             iter=1000, chains=4)
print(earnings_vary_si.sf1, pars = c("a1","a2", "sigma_y", "lp__"))
post1 <- extract(earnings_vary_si.sf1)

## plot on figure 13.3
age.label <- c("age 18-34", "age 35-49", "age 50-64")
eth.label <- c("BLACKS","HISPANICS","WHITES","OTHERS")
x <- height[ok] - mean(height[ok])
pushViewport(viewport(layout = grid.layout(1, 4)))
a.hat.M1 <- colMeans(post1$a)
b.hat.M1 <- colMeans(post1$b)

for (j in 1:4) {
  frame1 = data.frame(y1=y[eth.ok==j],x1=x[eth.ok==j])
  p1 <- ggplot(frame1, aes(x=x1, y=y1)) +
        geom_jitter(position = position_jitter(width = .05, height = 0)) +
        scale_x_continuous("Height (inches)") +
        scale_y_continuous("log(earnings)") +
        labs(title=eth.label[j]) +
        theme_bw() +
        geom_abline(aes(intercept = colMeans(post1$a)[j], slope=colMeans(post1$b)[j]), size = 0.25)
  for (s in 1:20)
    p1 <- p1 + geom_abline(intercept=post1$a[4000-s,j], slope=post1$b[4000-s,j],colour="grey10")
  print(p1, vp = viewport(layout.pos.row = 1, layout.pos.col = j))
}

## plot on figure 13.4
dev.new()
frame2 = data.frame(x1=colMeans(post1$a),y1=colMeans(post1$b))
p2 <- ggplot(frame2,aes(x=x1,y=y1)) +
      geom_point() +
      scale_y_continuous("Intercept") +
      scale_x_continuous("Slope") +
      theme_bw() +
print(p2)

 # plot on figure 13.5
dev.new()
frame3 = data.frame(x1=x,y1=y)
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

dataList.3 <- list(N=n, earn=earn[ok],height=x.centered,eth=eth[ok])
earnings_vary_si.sf2 <- stan(file='earnings_vary_si.stan', data=dataList.3,
                             iter=3000, chains=4)
print(earnings_vary_si.sf2)
post2 <- extract(earnings_vary_si.sf2)

pushViewport(viewport(layout = grid.layout(1, 4)))
a.hat.M2 <- colMeans(post2$a)
b.hat.M2 <- colMeans(post2$b)

for (j in 1:4) {
  frame1 = data.frame(y1=y[eth.ok==j],x1=x.centered[eth.ok==j])
  p4 <- ggplot(frame1, aes(x=x1, y=y1)) +
        geom_jitter(position = position_jitter(width = .05, height = 0)) +
        scale_x_continuous("Height (inches from mean)") +
        scale_y_continuous("log(earnings)") +
        labs(title=eth.label[j]) +
        theme_bw() +
        geom_abline(aes(intercept = colMeans(post2$a)[j], slope=colMeans(post2$b)[j]), size = 0.25)
  for (s in 1:20)
    p4 <- p4 + geom_abline(intercept=post2$a[4000-s,j], slope=post2$b[4000-s,j],colour="grey10")
  print(p4, vp = viewport(layout.pos.row = 1, layout.pos.col = j))
}

## Figure 13.7
dev.new()
frame5 = data.frame(x1=a.hat.M2,y1=b.hat.M2)
p5 <- ggplot(frame5,aes(x=x1,y=y1)) +
      geom_point() +
      scale_y_continuous("Intercept") +
      scale_x_continuous("Slope") +
      theme_bw() +
print(p5)
