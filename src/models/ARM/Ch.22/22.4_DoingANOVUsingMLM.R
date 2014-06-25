## Read the data
# Data are at http://www.stat.columbia.edu/~gelman/arm/examples/radon

# The R codes & data files should be saved in the same directory for
# the source command to work

library(rstan)
library(ggplot2)

srrs2 <- read.table ("srrs2.dat", header=T, sep=",")
mn <- srrs2$state=="MN"
radon <- srrs2$activity[mn]
log.radon <- log (ifelse (radon==0, .1, radon))
floor <- srrs2$floor[mn]       # 0 for basement, 1 for first floor
n <- length(radon)
y <- log.radon
x <- floor

# get county index variable
county.name <- as.vector(srrs2$county[mn])
uniq <- unique(county.name)
J <- length(uniq)
county <- rep (NA, J)
for (i in 1:J){
  county[county.name==uniq[i]] <- i
}

 # no predictors
ybarbar = mean(y)

sample.size <- as.vector (table (county))
sample.size.jittered <- sample.size*exp (runif (J, -.1, .1))
cty.mns = tapply(y,county,mean)
cty.vars = tapply(y,county,var)
cty.sds = mean(sqrt(cty.vars[!is.na(cty.vars)]))/sqrt(sample.size)
cty.sds.sep = sqrt(tapply(y,county,var)/sample.size)

dataList.1 <- list(N=length(y), y=y, county=county,J=85)
anova_radon_nopred.sf1 <- stan(file='anova_radon_nopred.stan', data=dataList.1,
                               iter=1000, chains=4)
print(anova_radon_nopred.sf1,pars = c("a","sigma_y","lp__","s_a","s_y"))

anova.df <- summary(anova_radon_nopred.sf1, c("s_y", "s_a"))$summary
anova.df <- data.frame(anova.df,
                       Source = factor(rownames(anova.df),
                                       levels = rownames(anova.df),
                                       labels = c("error", "county")),
                       df = with(dataList.1, c(N-J, J-1)))

# (close to) Figure 22.4
p <- ggplot(anova.df, aes(x = Source, y = X50., ymin = X2.5., ymax = X97.5.)) +
     geom_linerange()+ # 95% interval
     geom_linerange(aes(ymin = X25., ymax = X75.), size = 1)+ # 50% interval
     geom_point(size = 2)+ # Median
     scale_x_discrete("Source (df)",
                      labels = with(anova.df, paste0(Source, " (", df, ")"))) +
     scale_y_continuous("Est. sd of coefficients",
                        breaks = seq(0, .8, by=.2), # Breaks from 0 to 0.8
                        limits = c(0, max(anova.df$X97.5) + 0.1),
                        expand = c(0,0)) + # Remove y padding
     coord_flip() +
     theme_bw()
print(p)
