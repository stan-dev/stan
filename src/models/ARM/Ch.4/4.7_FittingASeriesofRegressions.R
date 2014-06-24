library(rstan)
library(ggplot2)

### Data file names

# nesYYYY.data.R, where YYYY == seq(1972, 2000, 4)

### Model

## Sampling over eight years & building a data frame for Figure 4.6

nes.post <- data.frame(c(), c(), c(), c(), c()) # empty data frame
years <- seq(1972, 2000, 4)
predictor <- c("Intercept", "Ideology", "Black", "Age.30.44", "Age.45.64",
               "Age.65.up", "Education", "Female", "Income")
predictor <- factor(predictor, levels = predictor)
for (i in years) {
    source(paste("nes", i, ".data.R", sep=""))
    data.list <- c("N", "partyid7", "real_ideo", "race_adj", "age_discrete",
                   "educ1", "gender", "income")
    sf <- stan(file='nes.stan', data=data.list, iter=1000, chains=4)
    beta.post <- extract(sf, "beta")$beta
    beta.mean <- colMeans(beta.post)
    beta.sd <- apply(beta.post, 2, sd)
    nes.post <- rbind(nes.post,
                      data.frame(year = rep(i, 9), predictor,
                                 Coefficient = beta.mean,
                                 lower = beta.mean - 0.67 * beta.sd,
                                 upper = beta.mean + 0.67 * beta.sd))
}

## Figure 4.6

p <- ggplot(nes.post) +
     geom_pointrange(aes(x = year, y = Coefficient, ymin = lower, ymax = upper)) +
     geom_hline(yintercept = 0, linetype = "dashed") +
     scale_x_continuous(breaks = seq(1972, 2000, 4),
                        labels = c("1972", "", "", "", "", "", "", "2000")) +
     theme_bw() +
     facet_wrap(~ predictor, scales = "free_y", ncol = 5)
print(p)
