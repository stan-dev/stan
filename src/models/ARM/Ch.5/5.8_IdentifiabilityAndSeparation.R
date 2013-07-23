library(rstan)
library(ggplot2)

## Generating the variables
x <- rnorm(60, mean =1, sd = 2)
y <- ifelse(x<2,0,1)

## Fit the model (y_x.stan)
## glm (y ~ x, family=binomial(link="logit"))
if (!file.exists("y_x.sm.RData")) {
    rt <- stanc("y_x.stan", model_name="y_x")
    y_x.sm <- stan_model(stanc_ret=rt)
    save(y_x.sm, file="y_x.sm.RData")
} else {
    load("y_x.sm.RData", verbose=TRUE)
}
dataList.1 <- list(N=length(x), y=y, x=x)
y_x.sf1 <- sampling(y_x.sm, dataList.1)
print(y_x.sf1)

## Plot
beta.post <- extract(y_x.sf1, "beta")$beta
b <- colMeans(beta.post)

frame = data.frame(y1=y,x1=x)
m <- ggplot(frame,aes(x=x1,y=y1))
m + geom_point() + scale_y_continuous("y") + scale_x_continuous("x", limits=c(-6,6)) + theme_bw() + stat_function(fun=function(x) 1.0 / (1 + exp(-(b[1]+b[2] * x))))
