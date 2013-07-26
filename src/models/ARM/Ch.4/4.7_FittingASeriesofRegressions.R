library(ggplot2)
library(rstan)
library(foreign)
## Figure 4.6

 # Read in the data

brdata <- read.dta("nes5200_processed_voters_realideo.dta",convert.factors=F)
# Data are at http://www.stat.columbia.edu/~gelman/arm/examples/nes

 # Clean the data
brdata <- brdata[is.na(brdata$black)==FALSE&is.na(brdata$female)==FALSE&is.na(brdata$educ1)==FALSE
&is.na(brdata$age)==FALSE&is.na(brdata$income)==FALSE&is.na(brdata$state)==FALSE,]
kept.cases <- 1952:2000
matched.cases <- match(brdata$year, kept.cases)
keep <- !is.na(matched.cases)
data <- brdata[keep,]
plotyear <- unique(sort(data$year))
year.new <- match(data$year,unique(data$year))
n.year <- length(unique(data$year))
income.new <-data$income-3
age.new <- (data$age-mean(data$age))/10
y <- data$rep_pres_intent
data <- cbind(data, year.new, income.new, age.new, y)
nes.year <- data[,"year"]
age.discrete <- as.numeric (cut (data[,"age"], c(0,29.5, 44.5, 64.5, 200)))
race.adj <- ifelse (data[,"race"]>=3, 1.5, data[,"race"])
data <- cbind (data, age.discrete, race.adj)

female <- data[,"gender"] - 1
black <- ifelse (data[,"race"]==2, 1, 0)
rvote <- ifelse (data[,"presvote"]==1, 0, ifelse(data[,"presvote"]==2, 1, NA))

region.codes <- c(3,4,4,3,4,4,1,1,5,3,3,4,4,2,2,2,2,3,3,1,1,1,2,2,3,2,4,2,4,1,1,4,1,3,2,2,3,4,1,
   1,3,2,3,3,4,1,3,4,1,2,4)
attach(data)

 # Regression & plot
if (!file.exists("nes.sm.RData")) {
    rt <- stanc("nes.stan", model_name="nes")
    nes.sm <- stan_model(stanc_ret=rt)
    save(nes.sm, file="nes.sm.RData")
} else {
    load("nes.sm.RData", verbose=TRUE)
}


regress.year <- function (yr) {
  this.year <- data[nes.year==yr,]
  ok <- !is.na(this.year$partyid7+this.year$real_ideo+this.year$race.adj+this.year$age.discrete+this.year$educ1+this.year$gender+this.year$income)
  dataList.1 <- list(N=length(this.year$partyid7[ok]), partyid7=this.year$partyid7[ok], real_ideo=this.year$real_ideo[ok],race_adj=this.year$race.adj[ok], age=this.year$age.discrete[ok],educ1=this.year$educ1[ok], gender=this.year$gender[ok], income=this.year$income[ok])
  nes.sf1 <- sampling(nes.sm, dataList.1)
  coefs <- extract(nes.sf1)$beta[,1:2]
}

summary <- array (NA, c(9,2,8))
for (yr in seq(1972,2000,4)){
  i <- (yr-1968)/4
  summary[,,i] <- regress.year(yr)
}

yrs <- array(yrs,c(72,1))
sum.n <- melt(summary)
sum.new <- subset(sum.n,Var2 == 1)
sum.new$Var3 <- factor(sum.new$Var3, levels=c("1","2","3","4","5","6","7","8","9"),labels=c("Intercept", "Ideology", "Black", "Age.30.44", "Age.45.64", "Age.65.up", 
   "Education", "Female", "Income"))
sum.new$yrs <- yrs
sum.new$sum1 <- melt(summary[,1,])$value
sum.new$sum2 <- melt(summary[,2,])$value
limits <- aes(ymax=sum1+.67*sum2,ymin=sum1-.67*sum2)
p <- ggplot(sum.new,aes(yrs,value)) + scale_x_continuous("Year") + scale_y_continuous("Coefficient") + theme_bw() + geom_hline(aes(yintercept=0))
p + geom_pointrange(limits)+ geom_point() + facet_wrap(~ Var3,ncol=4)
