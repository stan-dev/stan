library(rstan)
library(ggplot2)

ideo2 <- read.table ("ideo2.dat")
candidate.effects <- read.table ("candidate_effects.dat", row.names=1)

# Simple correction for incumbency advantage
incadv <- function (years){
  ifelse (years<46, .02,
    ifelse (years<66, .02 + .08*(years-46)/(66-46), .10))
}

# Regions of the country
region <- c(3,4,4,3,4,4,1,1,3,3,4,4,2,2,2,2,3,3,1,1,1,2,2,3,2,4,2,4,1,1,4,1,3,2,2,3,4,1,1,3,2,3,3,4,1,3,4,1,2,4)
south <- region==3


# add stuff to ideo2

normvote <- rep (NA, nrow(ideo2))
dum1 <- ideo2[,"dpvote"]
dum2 <- ideo2[,"stgj"]
dum3 <- (ideo2[,"year"]-44)/4
for (i in 1:length(normvote)){
  normvote[i] <- dum1[i] - candidate.effects[dum2[i],dum3[i]]
}
dvfix <- ideo2[,"dv"] - incadv(ideo2[,"year"])*ideo2[,"i2"]
dvpfix <- ideo2[,"dvp"] - incadv(ideo2[,"year"]-2)*ideo2[,"i1"]
ideo2 <- cbind(ideo2[,1:19], normvote, dvfix, dvpfix)
names(ideo2) <- c(names(ideo2)[1:19], "normvote", "dvfix", "dvpfix")

# impute normal vote from 2 years earlier:  ideo2

for (year in seq(62,94,4)){
  yr.cond <- ideo2[, "year"] == year
  normvote <- rep (NA, sum(yr.cond))
  dpvote <- rep (NA, sum(yr.cond))
  stgj <- ideo2[yr.cond,"stgj"]
  cd <- ideo2[yr.cond,"cd"]
  indexes <- (1:nrow(ideo2))[yr.cond]
  yr2.cond <- ideo2[, "year"] == year-2
  data2 <- ideo2[yr2.cond,c("stgj","cd","normvote","dpvote")]
  for (i in 1:sum(yr.cond)){
    cond <- data2[,"stgj"]==stgj[i] & data2[,"cd"]==cd[i]
    if (sum(cond)==1){
      normvote[i] <- data2[cond,c("normvote")]
      dpvote[i] <- data2[cond,c("dpvote")]
    }
  }
  ideo2[yr.cond,c("normvote")] <- normvote
  ideo2[yr.cond,c("dpvote")] <- dpvote
}

year <- 94
yr.cond <- ideo2[, "year"] == year
data <- ideo2[yr.cond,  ]
deminc.cond <- (data[, "dvp"] > 0.5) & (abs(data[,"i2"]) == 1)
repinc.cond <- (data[, "dvp"] < 0.5) & (abs(data[,"i2"]) == 1)
#
# fudge for 1992, 1994
if (year>=92) data[,"occup"] <- rep(0,nrow(data))
#
dum <- apply(is.na(data),1,sum)
ok <- dum==0 & !south[data[,"stgj"]]
attach(data)

 # Plot figure 10.8
frame1 = data.frame(x1=1-dvp[deminc.cond],y1=score1[deminc.cond])
frame2 = data.frame(x2=1-dvp[repinc.cond],y2=score1[repinc.cond])
p1 <- ggplot() +
      geom_point(data=frame1,aes(x=x1,y=y1),shape="x") +
      geom_point(data=frame2,aes(x=x2,y=y2),shape="o") +
      scale_y_continuous("(liberal)           ideology score       (conservative)") +
     scale_x_continuous("Republican's vote share") +
     theme_bw()
print(p1)

 # regression discontinuity analysis

x <- 1 - dvp
party <- ifelse (dvp<.5, 1, 0)

## Regression in the area near the discontinuity (ideo_two_pred.stan)
## lm (score1 ~ party + x, subset=overlap)

overlap <- (deminc.cond | repinc.cond) & dvp>.45 & dvp<.55 & !is.na(score1+party+x)
sc1 <- score1[overlap]
p1 <- party[overlap]
x1 <- x[overlap]
dataList.1 <- list(N=length(sc1), score1=sc1,party=p1,x=x1)
ideo_two_pred.sf1 <- stan(file='ideo_two_pred.stan', data=dataList.1,
                          iter=1000, chains=4)
print(ideo_two_pred.sf1)

## Regression fit to all data (ideo_two_pred.stan)
## lm (score1 ~ party + x, subset=incs)
incs <- (deminc.cond | repinc.cond) & !is.na(score1+party+x)
sc2 <- score1[incs]
p2 <- party[incs]
x2 <- x[incs]
dataList.2 <- list(N=length(sc2), score1=sc2,party=p2,x=x2)
ideo_two_pred.sf2 <- stan(file='ideo_two_pred.stan', data=dataList.2,
                          iter=1000, chains=4)
print(ideo_two_pred.sf2)

## Regression with interactions (ideo_interactions.stan)
## lm (score1 ~ party + x + party:x, subset=incs)

ideo_interactions.sf1 <- stan(file='ideo_interactions.stan', data=dataList.2,
                              iter=1000, chains=4)
print(ideo_interactions.sf1)

## Reparametrized regression (ideo_reparam.stan)
## lm (score1 ~ party + I(z*(party==0)) + I(z*(party==1)), subset=incs)

z <- x2 - 0.5
z.1 <- I(z*(p2==0))
z.2 <- I(z*(p2==1))
dataList.3 <- list(N=length(sc2), score1=sc2,party=p2,z1=z.1,z2=z.2)
ideo_reparam.sf1 <- stan(file='ideo_reparam.stan', data=dataList.3,
                         iter=1000, chains=4)
print(ideo_reparam.sf1)
