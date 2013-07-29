library(rstan)
library(ggplot2)
source("nes.data.R", echo = TRUE)    

 # Estimation (nes.stan)
 # glm (vote ~ income, family=binomial(link="logit"))

if (!exists("nes.sm")) {
    if (file.exists("nes.sm.RData")) {
        load("nes.sm.RData", verbose = TRUE)
    } else {
        rt <- stanc("nes.stan", model_name = "nes")
        nes.sm <- stan_model(stanc_ret = rt)
        save(nes.sm, file = "nes.sm.RData")
    }
}

dataList.1 <- c("N","income","vote")
nes.sf1 <- sampling(nes.sm, dataList.1)
print(nes.sf1, pars = c("beta", "lp__"))

fit1.post <- extract(nes.sf1)
beta.mean <- colMeans(fit1.post$beta)

 # Graph figure 5.1 (a)

frame1 = data.frame(income=income,vote=vote)
p1 <- ggplot(frame1,aes(x=income,y=vote)) +
      geom_jitter(position=position_jitter(height=.08,width=.4)) +
      scale_y_continuous("Pr(Republican Vote)",limits=c(-.01,1)) +
      scale_x_continuous("Income",limits=c(-2,8)) + theme_bw() +
      stat_smooth(method="glm",family="binomial",se=F,size=2,colour="black") +
      stat_function(fun=function(x) 1.0 / (1 + exp(-beta.mean[1] - beta.mean[2] * x)))
print(p1)

 # Graph figure 5.1 (b)

m2 <- "ggplot(frame1,aes(x=income,y=vote)) +
       scale_y_continuous('Pr(Republican Vote)',limits=c(-.01,1)) +
       scale_x_continuous('Income') +
       theme_bw() +
       stat_smooth(method='glm',family='binomial',se=F,colour='black') +
       geom_jitter(position=position_jitter(height=.08,width=.4))"
for (i in 1:20) {
  m2 <- paste(m2,"+ stat_function(aes(y=0),fun=function(x) 1.0 / (1 + exp(-fit1.post$beta[4000-",i,",1]-fit1.post$beta[4000-",i,",2]*x)),colour='grey')")
}
m2 <- paste(m2, "+ stat_function(fun=function(x) 1.0 / (1 + exp(-fit1.post$beta[1] - fit1.post$beta[2] * x)))")
eval(parse(text = m2))
