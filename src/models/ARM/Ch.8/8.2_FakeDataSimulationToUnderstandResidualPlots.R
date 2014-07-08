library(rstan)
library(ggplot2)
source("grades.data.R", echo = TRUE)    

## Estimate the model (grades.stan)
# lm (final ~ midterm)
dataList.1 <- c("N","midterm","final")
grades.sf1 <- stan(file='grades.stan', data=dataList.1, iter=1000, chains=4)
print(grades.sf1)

beta.post <- extract(grades.sf1, "beta")$beta
beta.mean <- colMeans(beta.post)

## Construct fitted values

n <- length(final)
X <- cbind (rep(1,n), midterm)
predicted <- X %*% beta.mean
resid <- final - predicted

## Simulate fake data & compute fitted values

a <- 65
b <- 0.7
sigma <- 15
y.fake <- a + b*midterm + rnorm (n, 0, sigma)

fake.sf1 <- sampling(grades.sm, dataList.1)
fake.post <- extract(fake.sf1, "beta")$beta
fake.mean <- colMeans(fake.post)
predicted.fake <- X %*% fake.mean
resid.fake <- y.fake - predicted.fake

## Plots figure 8.1

 # plot on the left

frame1 = data.frame(x1=predicted,x2=resid)
p1 <- ggplot(frame1,aes(x=x1,y=x2)) +
      geom_point() +
      scale_y_continuous("Residual") +
      scale_x_continuous("Predicted Value") +
      theme_bw() +
      geom_hline(yintercept=0,colour="grey") +
      labs(title="Residuals vs. Predicted Values")
print(p1)

 # plot on the right
dev.new()
frame2 = data.frame(x1=final,x2=resid)
p2 <- ggplot(frame2,aes(x=x1,y=x2)) +
      geom_point() +
      scale_y_continuous("Residual") +
      scale_x_continuous("Observed Value") +
      theme_bw() +
      geom_hline(yintercept=0,colour="grey") +
      labs(title="Residuals vs. Observed Values")
print(p2)

## Plots figure 8.2

 # plot on the left
dev.new()
frame3 = data.frame(x1=predicted.fake,x2=resid.fake)
p3 <- ggplot(frame3,aes(x=x1,y=x2)) +
      geom_point() +
      scale_y_continuous("Residual") +
      scale_x_continuous("Predicted Value") +
      theme_bw() +
      geom_hline(yintercept=0,colour="grey") +
      labs(title="Fake Data: Residuals vs. Predicted Values")
print(p3)

 # plot on the right
dev.new()
frame4 = data.frame(x1=y.fake,x2=resid.fake)
p4 <- ggplot(frame4,aes(x=x1,y=x2)) +
      geom_point() +
      scale_y_continuous("Residual") +
      scale_x_continuous("Observed Value") +
      theme_bw() +
      geom_hline(yintercept=0,colour="grey") +
      labs(title="Fake Data: Residuals vs. Observed Values")
print(p4)
