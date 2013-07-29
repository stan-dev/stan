library(rstan)
library(ggplot2)
## A simple example of discrete predictive simulations

n.girls <- rbinom (1, 400, .488)
print (n.girls)

n.sims <- 1000
n.girls <- rep (NA, n.sims)
for (s in 1:n.sims){
  n.girls[s] <- rbinom (1, 400, .488)
}

frame1 = data.frame(x1=n.girls)
m <- ggplot(frame1,aes(x=x1))  + scale_x_continuous("Number of Girls")
m + geom_histogram(colour = "black", fill = "white",binwidth=5) + theme_bw()

## Accounting for twins

birth.type <- sample (c("fraternal twin", "identical twin", "single birth"),
  size=400, replace=TRUE, prob=c(1/25, 1/300, 1 - 1/25 - 1/300))
girls <- rep (NA, 400)
for (i in 1:400){
  if (birth.type[i]=="single birth"){
   girls[i] <- rbinom (1, 1, .488)}
  else if (birth.type[i]=="identical twin"){
   girls[i] <- 2*rbinom (1, 1, .495)}
  else if (birth.type[i]=="fraternal twin"){
   girls[i] <- rbinom (1, 2, .495)}
}
n.girls <- sum (girls)

 # putting in a loop

n.sims <- 1000
n.girls <- rep (NA, n.sims)
for (s in 1:n.sims){
 birth.type <- sample (c("fraternal twin", "identical twin", "single birth"),
   size=400, replace=TRUE, prob=c(1/25, 1/300, 1 - 1/25 - 1/300))
 girls <- rep (NA, 400)
 for (i in 1:400){
  if (birth.type[i]=="single birth"){
   girls[i] <- rbinom (1, 1, .488)}
  else if (birth.type[i]=="identical twin"){
   girls[i] <- 2*rbinom (1, 1, .495)}
  else if (birth.type[i]=="fraternal twin"){
   girls[i] <- rbinom (1, 2, .495)}
}
n.girls[s] <- sum (girls)
}

## A simple example of continuos predictive simulations

woman <- rbinom (10, 1, .52)
height <- ifelse (woman==0, rnorm (10, 69.1, 2.9), rnorm (10, 64.5, 2.7))
avg.height <- mean (height)
print(avg.height)

 # simulation & Figure 7.1

n.sims <- 1000
avg.height <- rep (NA, n.sims)
for (s in 1:n.sims){
  sex <- rbinom (10, 1, .52)
  height <- ifelse (sex==0, rnorm (10, 69.1, 2.9), rnorm (10, 64.5, 2.7))
  avg.height[s] <- mean (height)
}

frame2 = data.frame(x1=avg.height)
p1 <- ggplot(frame2,aes(x=x1))  +
      scale_x_continuous("Average height of 10 adults") +
      geom_histogram(colour = "black", fill = "white") +
      theme_bw()
print(p1)

 # simulation for the maximum height

n.sims <- 1000
max.height <- rep (NA, n.sims)
for (s in 1:n.sims){
  sex <- rbinom (10, 1, .52)
  height <- ifelse (sex==0, rnorm (10, 69.1, 2.9), rnorm (10, 64.5, 2.7))
  max.height[s] <- max (height)
}

dev.new()
frame3 = data.frame(x1=max.height)
p2 <- ggplot(frame3,aes(x=x1))  +
      scale_x_continuous("Maximum height of 10 adults") +
      geom_histogram(colour = "black", fill = "white") +
      theme_bw()
print(p2)

## Simulation using custom-made functions

Height.sim <- function (n.adults){
  sex <- rbinom (n.adults, 1, .52)
  height <- ifelse (sex==0, rnorm (10, 69.1, 2.9), rnorm (10, 64.5, 2.7))
  return (mean(height))
}

frame4 = data.frame(x1=avg.height)
p3 <- ggplot(frame4,aes(x=x1))  +
      scale_x_continuous("Average height of 10 adults") +
      geom_histogram(colour = "black", fill = "white") +
      theme_bw()
print(p3)
