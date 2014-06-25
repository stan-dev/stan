library(rstan)
library(ggplot2)

## Data

source("radon.data.R", echo = TRUE)

## Classical complete pooling regression
lm.pooled <- lm (y ~ x)
summary(lm.pooled)

## Classical no pooling regression

# with the constant term
lm.unpooled.0 <- lm (y ~ x + factor(county))
summary(lm.unpooled.0)

# without the constant term
lm.unpooled <- lm (y ~ x + factor(county) - 1)
summary(lm.unpooled)

## Call Stan from R
radon.data <- c("N", "J", "y", "x", "county")

# with 10 iterations
radon.1.sf <- stan(file='radon.1.stan', data=radon.data, iter = 10, chains=4)

# with 500 iterations
radon.1.sf <- stan(file='radon.1.stan', data=radon.data, iter = 500, chains=4)
plot(radon.1.sf)   # to get a plot similar to Figure 16.1
print(radon.1.sf)  # to display the results in the R console

## Summarizing classical and multilevel inferences graphically

# choose countries
display8 <- c(36, 1, 35, 21, 14, 71, 61, 70)  # counties to be displayed
radon.ggdf <- subset(data.frame(y, x, county), county %in% display8)
radon.ggdf$county.name <- county_name[radon.ggdf$county]
radon.ggdf$county.name <- factor(radon.ggdf$county.name,
                                 levels=county_name[display8])

# pull out parameter estimates from classical fits
a.pooled <- coef(lm.pooled)[1]                  # complete-pooling intercept
radon.ggdf$a.pooled <- a.pooled
radon.ggdf$b.pooled <- coef(lm.pooled)[2]       # complete-pooling slope
a.nopooled <- coef(lm.unpooled)[2:(J+1)]        # no-pooling vector of intercepts
radon.ggdf$a.nopooled <- a.nopooled[radon.ggdf$county]
radon.ggdf$b.nopooled <- coef(lm.unpooled)[1]   # no-pooling slope

# compute medians from Stan fit
sims <- extract(radon.1.sf)
a <- sims$a
b <- sims$b
a.multilevel <- rep(NA, J)
for (j in 1:J)
    a.multilevel[j] <- median(a[,j])
radon.ggdf$a.multilevel <- a.multilevel[radon.ggdf$county]
radon.ggdf$b.multilevel <- median(b)

# make the plot in Figure 12.4

p1 <- ggplot(radon.ggdf, aes(x, y)) +
    geom_jitter(position = position_jitter(width = 0.05, height = 0)) +
    geom_abline(aes(intercept = a.pooled, slope = b.pooled), linetype = "dashed") +
    geom_abline(aes(intercept = a.nopooled, slope = b.nopooled), size = 0.25) +
    geom_abline(aes(intercept = a.multilevel, slope = b.multilevel)) +
    scale_x_continuous("floor", breaks=c(0,1), labels=c("0", "1")) +
    ylab("log radon level") +
    facet_wrap(~ county.name, ncol = 4)
print(p1)

# displaying estimates and uncertainties and plot in Figure 12.3b

a.sd <- rep(NA, J)
for (j in 1:J)
    a.sd[j] <- sd(a[,j])
estimates.ggdf <- data.frame(sample.size = as.vector(table(county)),
                             a.multilevel, a.sd)
dev.new()
p2 <- ggplot(estimates.ggdf, aes(x = sample.size, y = a.multilevel)) +
    geom_pointrange(aes(ymin = a.multilevel - a.sd, ymax = a.multilevel + a.sd),
                    position = position_jitter(width = 0.1, height = 0)) +
    geom_hline(yintercept = a.pooled, size = 0.5) +
    scale_x_continuous("sample size in country j",
                       trans = "log", breaks = c(1, 3, 10, 30, 100)) +
    scale_y_continuous(expression(paste("intercept, ", alpha[j],
                                        "   (multilevel inference)")),
                       limits = c(0, 3))
print(p2)
