library(rstan)
library(ggplot2)

### Data

source("wells.data.R", echo = TRUE)

### Model: switched ~ c_dist100 + c_arsenic + c_educ4 + c_dist100:c_arsenic
###                   + c_dist100:c_educ4 + c_arsenic:c_educ4
### c_dist100 <- (dist - mean(dist)) / 100
### c_arsenic <- arsenic - mean(arsenic)
### c_educ4   <- (educ - mean(educ)) / 4

data.list <- c("N", "switched", "dist", "arsenic", "educ")
wells_predicted.sf <- stan(file='wells_predicted.stan', data=data.list,
                           iter=1000, chains=4)
print(wells_predicted.sf, pars = c("beta", "lp__"))

## Residual Plot (Figure 5.13 (a))

prob.pred.1 <- colMeans(extract(wells_predicted.sf, "pred")$pred)
wells_resid.ggdf.1 <- data.frame(prob = prob.pred.1, resid = switched - prob.pred.1)
p1 <- ggplot(wells_resid.ggdf.1, aes(prob, resid)) + geom_point() +
    scale_x_continuous("Estimated Pr(switching)", limits = c(0, 1),
                       breaks = seq(0, 1, 0.2)) +
    scale_y_continuous("Observed - estimated", limits = c(-1, 1),
                       breaks = seq(-1, 1, 0.5)) +
    ggtitle("Residual plot")
print(p1)

## Binned residual plot

# Defining binned residuals

binned.resids <- function (x, y, nclass = sqrt(length(x))) {
    breaks.index <- floor(length(x) * (1:(nclass-1)) / nclass)
    breaks <- c (-Inf, sort(x)[breaks.index], Inf)
    output <- NULL
    xbreaks <- NULL
    x.binned <- as.numeric(cut (x, breaks))
    for (i in 1:nclass) {
        items <- (1:length(x))[x.binned == i]
        x.range <- range(x[items])
        xbar <- mean(x[items])
        ybar <- mean(y[items])
        n <- length(items)
        sdev <- sd(y[items])
        output <- rbind(output, c(xbar, ybar, n, x.range, 2 * sdev / sqrt(n)))
    }
    colnames (output) <- c("xbar", "ybar", "n", "x.lo", "x.hi", "2se")
    return(list(binned = output, xbreaks = xbreaks))
}

# Binned residuals vs. estimated probability of switched (Figure 5.13 (b))

dev.new()
br <- binned.resids(prob.pred.1, switched - prob.pred.1, nclass = 40)$binned
binned.ggdf.1 <- data.frame(x = br[,1], y = br[,2], disp = br[,6])
p2 <- ggplot(binned.ggdf.1, aes(x, y)) +
    geom_point() +
    geom_line(aes(x = x, y = disp), color = "gray") +
    geom_line(aes(x = x, y = - disp), color = "gray") +
    geom_hline(yintercept = 0, color = "gray") +
    scale_x_continuous("Estimated Pr(switching)", breaks = seq(0.3, 0.9, 0.1)) +
    scale_y_continuous("Average residual") +
    ggtitle("Binned residual plot")
print(p2)

## Plot of binned residuals vs. inputs of interest

# distance (Figure 5.13 (a))
dev.new()
br.dist <- binned.resids(dist, switched - prob.pred.1, nclass = 40)$binned
binned.ggdf.2 <- data.frame(x = br.dist[,1], y = br.dist[,2], disp = br.dist[,6])
p3 <- ggplot(binned.ggdf.2, aes(x, y)) +
    geom_point() +
    geom_line(aes(x = x, y = disp), color = "gray") +
    geom_line(aes(x = x, y = - disp), color = "gray") +
    geom_hline(yintercept = 0, color = "gray") +
    scale_x_continuous("Distance to nearest safe well", breaks = seq(0, 150, 50)) +
    scale_y_continuous("Average residual") +
    ggtitle("Binned residual plot")
print(p3)

# arsenic (Figure 5.13 (b))
dev.new()
br.as <- binned.resids(arsenic, switched - prob.pred.1, nclass = 40)$binned
binned.ggdf.3 <- data.frame(x = br.as[,1], y = br.as[,2], disp = br.as[,6])
p4 <- ggplot(binned.ggdf.3, aes(x, y)) +
    geom_point() +
    geom_line(aes(x = x, y = disp), color = "gray") +
    geom_line(aes(x = x, y = - disp), color = "gray") +
    geom_hline(yintercept = 0, color = "gray") +
    scale_x_continuous("Arsenic level", breaks = seq(0, 5)) +
    scale_y_continuous("Average residual") +
    ggtitle("Binned residual plot")
print(p4)

### Log transformation: switched ~ c_dist100 + c_log_arsenic + c_educ4
###                                + c_dist100:c_log_arsenic + c_dist100:c_educ4
###                                + c_log_arsenic:c_educ4
### c_log_arsenic <- log(arsenic) - mean(log(arsenic))
wells_predicted_log.sf <- stan(file='wells_predicted_log.stan',
                               data=data.list,
                               iter=1000, chains=4)
print(wells_predicted_log.sf, pars = c("beta", "lp__"))

beta.post <- extract(wells_predicted_log.sf, "beta")$beta
beta.mean <- colMeans(beta.post)

## Graph for log model (Figure 5.15 (a))

dev.new()
p5 <- ggplot(data.frame(switched, arsenic), aes(arsenic, switched)) +
    geom_jitter(position = position_jitter(width = 0.2, height = 0.01)) +
    stat_function(fun = function(x)
                  1 / (1 + exp(
                      - cbind(1, 0, log(x), mean(educ / 4), 0 * log(x),
                              0 * mean(educ / 4), log(x) * mean(educ / 4))
                      %*% beta.mean))) +
    stat_function(fun = function(x)
                  1 / (1 + exp(
                      - cbind(1, 0.5, log(x), mean(educ / 4), 0.5 * log(x),
                              0.5 * mean(educ / 4), log(x) * mean(educ / 4))
                      %*% beta.mean))) +
    annotate("text", x = c(1.7,2.5), y = c(0.82, 0.66),
             label = c("if dist = 0", "if dist = 50"), size = 4) +
    scale_x_continuous("Arsenic concentration in well water",
                       breaks = seq(from = 0, by = 2, length.out = 5)) +
    scale_y_continuous("Pr(switching)", breaks = seq(0, 1, 0.2))
print(p5)

## Graph of binned residuals for log model (Figure 5.15 (b))

dev.new()
prob.pred.2 <- colMeans(extract(wells_predicted_log.sf, "pred")$pred)
br.log <- binned.resids(arsenic, switched - prob.pred.2, nclass = 40)$binned
binned.ggdf.2 <- data.frame(x = br.log[,1], y = br.log[,2], disp = br.log[,6])
p6 <- ggplot(binned.ggdf.2, aes(x, y)) +
    geom_point() +
    geom_line(aes(x = x, y = disp), color = "gray") +
    geom_line(aes(x = x, y = - disp), color = "gray") +
    geom_hline(yintercept = 0, color = "gray") +
    scale_x_continuous("Arsenic level", limits = c(0, max(br.2[,1])),
                       breaks = seq(0, 5)) +
    scale_y_continuous("Average residual") +
    ggtitle("Binned residual plot\nfor model with log(arsenic)")
print(p6)

### Error rate

error.rate <- mean((prob.pred.2 > 0.5 & switched == 0) |
                   (prob.pred.2 < 0.5 & switched == 1))
error.rate
