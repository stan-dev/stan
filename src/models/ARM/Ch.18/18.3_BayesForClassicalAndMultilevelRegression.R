library(rstan)
library(grid)
library(ggplot2)

### Data

source("radon.data.R", echo = TRUE)
radon.data <- c("N", "J", "y", "x", "county", "u")

### Complete pooling regression

radon.pooling.sf <- stan(file='radon.pooling.stan', data=radon.data,
                         iter=1000, chains=4)
print(radon.pooling.sf)

### No pooling regression

radon.nopooling.sf <- stan(file='radon.nopooling.stan', data=radon.data,
                           iter=1000, chains=4)
print(radon.nopooling.sf)

### Multilevel model with no group-level predictors

radon.1.sf <- stan(file='radon.1.stan', data=radon.data, iter=1000, chains=4)
print(radon.1.sf)

## Plot Figure 18.5

display <- c(36, 35, 14, 61)  # counties to display

a.unpooled <- extract(radon.nopooling.sf, "a")$a
mu.unpooled <- colMeans(a.unpooled)[display]
se.unpooled <- apply(a.unpooled, 2, sd)[display]

sims <- extract(radon.1.sf)
mu.a <- mean(sims$mu_a)
sigma.a <- mean(sims$sigma_a)
a.j <- colMeans(sims$a[,display])
a.j.sigma <- apply(sims$a[,display], 2, sd)

pushViewport(viewport(layout = grid.layout(3, 4)))
p <- ggplot(data.frame(x = c(-0.5, 5)), aes(x)) +
    scale_x_continuous(expression(alpha[j]), breaks = c(0, 2, 4)) +
    scale_y_continuous("", breaks = NULL) +
    theme(plot.title = element_text(size = 7, face = "bold"))
    
for (j in 1:4) {
    p1 <- p +
        stat_function(fun = dnorm,
                      args = list(mean = mu.unpooled[j], sd = se.unpooled[j])) +
        ggtitle(paste(county_name[display[j]], ": likelihood", sep = ''))
    print(p1, vp = viewport(layout.pos.row = 1, layout.pos.col = j))
}

for (j in 1:4) {
    p1 <- p + stat_function(fun = dnorm,
                            args = list(mean = mu.a, sd = sigma.a)) +
        ggtitle("prior dist.")
    print(p1, vp = viewport(layout.pos.row = 2, layout.pos.col = j))
}

for (j in 1:4) {
    p1 <- p + stat_function(fun = dnorm,
                            args = list(mean = a.j[j], sd = a.j.sigma[j])) +
        ggtitle("posterior dist.")
    print(p1, vp = viewport(layout.pos.row = 3, layout.pos.col = j))
}

### Multilevel model with group-level predictors

radon.2.sf <- stan(file='radon.2.stan', data=radon.data, iter=1000, chains=4)
print(radon.2.sf)

## Plot Figure 18.6

dev.new()

sims <- extract(radon.2.sf)
mu.a <- mean(sims$g_0) + mean(sims$g_1) * u[display]
sigma.a <- mean(sims$sigma_a)
a.j <- colMeans(sims$a[,display])
a.j.sigma <- apply(sims$a[,display], 2, sd)

pushViewport(viewport(layout = grid.layout(3, 4)))
p <- ggplot(data.frame(x = c(-0.5, 5)), aes(x)) +
    scale_x_continuous(expression(alpha[j]), breaks = c(0, 2, 4)) +
    scale_y_continuous("", breaks = NULL) +
    theme(plot.title = element_text(size = 7, face = "bold"))
    
for (j in 1:4) {
    p1 <- p +
        stat_function(fun = dnorm,
                      args = list(mean = mu.unpooled[j], sd = se.unpooled[j])) +
        ggtitle(paste(county_name[display[j]], ": likelihood", sep = ''))
    print(p1, vp = viewport(layout.pos.row = 1, layout.pos.col = j))
}

for (j in 1:4) {
    p1 <- p + stat_function(fun = dnorm,
                            args = list(mean = mu.a[j], sd = sigma.a)) +
        ggtitle("prior dist.")
    print(p1, vp = viewport(layout.pos.row = 2, layout.pos.col = j))
}

for (j in 1:4) {
    p1 <- p + stat_function(fun = dnorm,
                            args = list(mean = a.j[j], sd = a.j.sigma[j])) +
        ggtitle("posterior dist.")
    print(p1, vp = viewport(layout.pos.row = 3, layout.pos.col = j))
}
