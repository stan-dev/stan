library(rstan)
library(ggplot2)
## Read the data
# Data are at http://www.stat.columbia.edu/~gelman/arm/examples/radon

# The R codes & data files should be saved in the same directory for
# the source command to work

source("12.3_PartialPoolingWithPredictors.R") # where variables were defined

## Get the county-level predictor
srrs2.fips <- srrs2$stfips*1000 + srrs2$cntyfips
cty <- read.table ("cty.dat", header=T, sep=",")
usa.fips <- 1000*cty[,"stfips"] + cty[,"ctfips"]
usa.rows <- match (unique(srrs2.fips[mn]), usa.fips)
uranium <- cty[usa.rows,"Uppm"]
u <- log (uranium)

## Varying-intercept model w/ group-level predictors
u.full <- u[county]
if (!exists("radon_group.sm")) {
    if (file.exists("radon_group.sm.RData")) {
        load("radon_group.sm.RData", verbose = TRUE)
    } else {
        rt <- stanc("radon_group.stan", model_name = "radon_group")
        radon_group.sm <- stan_model(stanc_ret = rt)
        save(radon_group.sm, file = "radon_group.sm.RData")
    }
}
dataList.3 <- list(N=length(y), y=y,x=x,county=county,u=u.full)
radon_group.sf1 <- sampling(radon_group.sm, dataList.3)
print(radon_group.sf1)
post <- extract(radon_group.sf1)

## Plots on Figure 12.5 FIXME: model too slow
M2 <- lmer (y ~ x + u.full + (1 | county))
M1 <- lmer (y ~ x + (1 | county))
a.hat.M1 <- fixef(M1)[1] + ranef(M1)$county                
b.hat.M1 <- fixef(M1)[2]
a.hat.M2 <- fixef(M2)[1] + fixef(M2)[3]*u + ranef(M2)$county
b.hat.M2 <- fixef(M2)[2]
a.hat.M1 <- melt(a.hat.M1)
a.hat.M2 <- melt(a.hat.M2)
b.hat.M1 <- melt(b.hat.M1)
b.hat.M2 <- melt(b.hat.M2)

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
radon8.data$m1.int <- a.hat.M1$value[radon8.data$county]
radon8.data$m1.slope <- rep(b.hat.M1$value,209)
radon8.data$m2.int <- a.hat.M2$value[radon8.data$county]
radon8.data$m2.slope <- rep(b.hat.M2$value,209)

p1 <- ggplot(radon8.data, aes(x.jitter, y)) +
     geom_jitter(position = position_jitter(width = .05, height = 0)) +
     scale_x_continuous(breaks=c(0,1), labels=c("0", "1")) +
     geom_abline(aes(intercept = m1.int, slope = m1.slope), size = 0.25,colour="grey10") +
     geom_abline(aes(intercept = m2.int,slope=m2.slope), size = 0.25) +
     facet_wrap(~ county.name, ncol = 4)
print(p1)

# Plot of ests & se's vs. county uranium (Figure 12.6)
a.se.M2 <- se.coef(M2)$county
a.se.M2 <- melt(a.se.M2)
dev.new()

frame1 = data.frame(x1=u,y1=a.hat.M2$value)
limits <- aes(ymax=a.hat.M2$value+a.se.M2$value,ymin=a.hat.M2$value-a.se.M2$value)
p2 <- ggplot(frame1,aes(x=x1,y=y1)) +
      geom_point() +
      theme_bw() +
      geom_pointrange(limits) +
      geom_abline(aes(intercept=fixef(M2)["(Intercept)"],slope=fixef(M2)["u.full"]))
print(p2)
