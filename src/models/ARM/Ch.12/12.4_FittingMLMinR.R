library(rstan)
library(ggplot2)
# Data are at http://www.stat.columbia.edu/~gelman/arm/examples/radon

# The R codes & data files should be saved in the same directory for
# the source command to work

source("12.3_Partial pooling with predictors.R") # where variables were defined

## Varying-intercept model w/ no predictors

dataList.1 <- list(N=length(y), y=y, county=county)
radon_intercept.sf1 <- sampling(radon_intercept.sm, dataList.1)
print(radon_intercept.sf1)

## Including x as a predictor

dataList.2 <- list(N=length(y), y=y,x=x,county=county)
radon_no_pool.sf1 <- sampling(radon_no_pool.sm, dataList.2)
print(radon_no_pool.sf1)

M1 <- extract(radon_no_pool.sf1)
M1.factor <- colMeans(M1$factor)
M1.beta <- mean(M1$beta)

  # 95% CI for the slope
M1.beta + c(-2,2) * sd(M1$beta[,1]) / sqrt(4000)

  # 95% CI for the intercept in county 26
M1.factor[26] + c(-2,2) * sd(M1$factor[,26])/sqrt(4000)

# to plot Figure 12.4
M1 <- lmer (y ~ x + (1 | county))
a.hat.M1 <- coef(M1)$county[,1]                # 1st column is the intercept
b.hat.M1 <- coef(M1)$county[,2]                # 2nd element is the slope

x.jitter <- x + runif(n,-.05,.05)
display8 <- c (36, 1, 35, 21, 14, 71, 61, 70)  # counties to be displayed
y.range <- range (y[!is.na(match(county,display8))])

radon.data <- data.frame(y, x.jitter, county)
radon8.data <- subset(radon.data, county %in% display8)
radon8.data$county.name <- radon8.data$county
radon8.data$county.name <- factor(radon8.data$county.name,levels=c("36","1","35","21","14","71","61","70"),
                                  labels=c("LAC QUI PARLE", "AITKIN", "KOOCHICHING",
                                      "DOUGLAS", "CLAY", "STEARNS", "RAMSEY",
                                      "ST LOUIS"))
radon8.data$pooled.int <- pooled[1]
radon8.data$pooled.slope <- pooled[2]
radon8.data$unpooled.int <- unpooled[radon8.data$county]
radon8.data$unpooled.slope <- mean(post.unpooled$beta)
radon8.data$coef.int <- a.hat.M1[radon8.data$county]
radon8.data$coef.slope <- b.hat.M1[radon8.data$county]

p1 <- ggplot(radon8.data, aes(x.jitter, y)) +
    geom_jitter(position = position_jitter(width = .05, height = 0)) +
    scale_x_continuous(breaks=c(0,1), labels=c("0", "1")) +
    geom_abline(aes(intercept = pooled.int, slope = pooled.slope), linetype = "dashed") +
    geom_abline(aes(intercept = unpooled.int, slope = unpooled.slope), size = 0.25) +
    geom_abline(aes(intercept = coef.int, slope = coef.slope), size = 0.25) +
    facet_wrap(~ county.name, ncol = 4)
print(p1)

## Multilevel model ests vs. sample size (plot on the right on figure 12.3)
dev.new()
a.se.M1 <- se.coef(M1)$county
sample.size <- as.vector (table (county))
sample.size.jittered <- sample.size*exp (runif (J, -.1, .1))
max1 <- a.hat.M1+a.se.M1
min1 <- a.hat.M1-a.se.M1
frame3 = data.frame(x1=sample.size.jittered,y1=unpooled,max1=max1,min1=min1)
p2 <- ggplot(frame3,aes(x=x1,y=y1,ymin= min1,ymax=max1)) +
      geom_point() +
      scale_y_continuous("estimated intercept alpha (no pooling)") +
      scale_x_log10("Sample Size in County j") +
      theme_bw() +
      geom_linerange() +
      geom_hline(aes(yintercept=pooled[1]))
print(p2)
