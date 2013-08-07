library(rstan)
library(ggplot2)
library(gridBase)
electric <- read.table ("electric.dat", header=T)
attach(electric)
## Plot of the raw data (Figure 9.4)
  
  electric$Grade <- factor(electric$Grade, levels=c('1','2','3','4'), labels=c('Grade 1', 'Grade 2', 'Grade 3', 'Grade 4'))
  frame = data.frame(x1=control.Posttest,Grade=electric$Grade)
  p1 <- ggplot(frame,aes(x1)) +
        geom_histogram(colour = "black", fill = "white", binwidth=5) +
        theme_bw() +
        facet_grid(Grade ~.) +
        theme(strip.text.y = element_blank(),axis.title.y = element_blank(),axis.title.x=element_blank(),strip.background = element_blank()) +
        labs(title='Test Scores in Control Classes') 

  frame2 = data.frame(x1=treated.Posttest,Grade=electric$Grade)
  p2 <- ggplot(frame2,aes(x1)) +
        geom_histogram(colour = "black", fill = "white", binwidth=5) +
        theme_bw() +
        facet_grid(Grade ~.) +
        theme(axis.title.y = element_blank(),axis.ticks.y = element_blank(),axis.text.y=element_blank(),axis.title.x=element_blank()) +
        labs(title='Test Scores in Treated Classes') 

pushViewport(viewport(layout = grid.layout(1, 2)))
print(p1, vp = viewport(layout.pos.row = 1, layout.pos.col = 1))
print(p2, vp = viewport(layout.pos.row = 1, layout.pos.col = 2))

## Basic analysis of a completely randomized experiment
electric <- read.table ("electric.dat", header=T)
attach(electric)

post.test <- c (treated.Posttest, control.Posttest)
pre.test <- c (treated.Pretest, control.Pretest)
grade <- rep (electric$Grade, 2)
treatment <- rep (c(1,0), rep(length(treated.Posttest),2))
n <- length (post.test)

if (!exists("electric_one_pred.sm")) {
    if (file.exists("electric_one_pred.sm.RData")) {
        load("electric_one_pred.sm.RData", verbose = TRUE)
    } else {
        rt <- stanc("electric_one_pred.stan", model_name = "electric_one_pred")
        electric_one_pred.sm <- stan_model(stanc_ret = rt)
        save(electric_one_pred.sm, file = "electric_one_pred.sm.RData")
    }
}

est1 <- rep(NA,4)
se1 <- rep(NA,4)
for (k in 1:4) {
post.test1 <- post.test[grade==k]
treatment1 <- treatment[grade==k]
ok <- !is.na(post.test1+treatment1)
post.test2 <- post.test1[ok]
treatment2 <- treatment1[ok]
N1 <- length(post.test2)

dataList.1 <- list(N=N1, post_test=post.test2, treatment=treatment2)
electric_one_pred.sf1 <- sampling(electric_one_pred.sm, dataList.1)
print(electric_one_pred.sf1)
beta.post <- extract(electric_one_pred.sf1, "beta")$beta
est1[k] <- colMeans(beta.post)[2]
se1[k] <- sd(beta.post[,2])
}

## Plot of the regression results (Figure 9.5)
if (!exists("electric_multi_preds.sm")) {
    if (file.exists("electric_multi_preds.sm.RData")) {
        load("electric_multi_preds.sm.RData", verbose = TRUE)
    } else {
        rt <- stanc("electric_multi_preds.stan", model_name = "electric_multi_preds")
        electric_multi_preds.sm <- stan_model(stanc_ret = rt)
        save(electric_multi_preds.sm, file = "electric_multi_preds.sm.RData")
    }
}

est2 <- rep(NA,4)
se2 <- rep(NA,4)
for (k in 1:4) {
post.test1 <- post.test[grade==k]
pre.test1 <- pre.test[grade==k]
treatment1 <- treatment[grade==k]
ok <- !is.na(post.test1+treatment1+pre.test1)
post.test2 <- post.test1[ok]
pre.test2 <- pre.test1[ok]
treatment2 <- treatment1[ok]
N1 <- length(post.test2)

dataList.1 <- list(N=N1, post_test=post.test2, treatment=treatment2,pre_test=pre.test2)
electric_multi_preds.sf1 <- sampling(electric_multi_preds.sm, dataList.1)
print(electric_multi_preds.sf1)
beta.post <- extract(electric_multi_preds.sf1, "beta")$beta
est2[k] <- colMeans(beta.post)[2]
se2[k] <- sd(beta.post[,2])
}

#graphs on Figure 9.5

frame = data.frame(Grade=4:1,x1=est1,x2=est2,se1=se1,se2=se2)
dev.new()
p4 <- ggplot(frame, aes(x=x1,y=Grade)) +
      geom_point(size=3) +
      theme_bw() +
      labs(title="Regression on Treatment Indicator") +
      geom_segment(aes(x=x1-se1,y=Grade,xend=x1+se1,yend=Grade),size=2) + 
      geom_segment(aes(x=x1-2 * se1,y=Grade,xend=x1+2*se1,yend=Grade)) +
      geom_vline(xintercept=0,linetype="dotted") +
      scale_x_continuous("Effect of the Electric Company TV Show")

p5 <- ggplot(frame, aes(x=x2,y=Grade)) +
      geom_point(size=3) +
      theme_bw() +
      labs(title="Regression on Treatment Indicator Controlling for Pre-test") +
      geom_segment(aes(x=x2-se2,y=Grade,xend=x2+se2,yend=Grade),size=2) + 
      geom_segment(aes(x=x2-2 * se2,y=Grade,xend=x2+2*se2,yend=Grade)) +
      geom_vline(xintercept=0,linetype="dotted") +
      scale_x_continuous("Effect of the Electric Company TV Show")

pushViewport(viewport(layout = grid.layout(1, 2)))
print(p4, vp = viewport(layout.pos.row = 1, layout.pos.col = 1))
print(p5, vp = viewport(layout.pos.row = 1, layout.pos.col = 2))


## Controlling for pre-treatment predictors (Figure 9.6)
electric <- read.table ("electric.dat", header=T)
attach(electric)
dev.new()
pushViewport(viewport(layout = grid.layout(1, 4)))
for (j in 1:4){
  ok <- electric$Grade==j & !is.na(treated.Posttest+treated.Pretest+control.Pretest+control.Posttest)
  pret <- c (treated.Pretest[ok], control.Pretest[ok])
  postt <-c (treated.Posttest[ok], control.Posttest[ok])
  t <- rep (c(1,0),rep(sum(ok),2))
  dataList.1 <- list(N=length(pret), post_test=postt, pre_test=pret,treatment=t)
  electric_multi_preds.sf <- sampling(electric_multi_preds.sm, dataList.1)
  beta.post <- extract(electric_multi_preds.sf, "beta")$beta
  beta.mean <- colMeans(beta.post)

  frame1 = data.frame(x1=treated.Pretest[ok],y1=treated.Posttest[ok])
  frame2 = data.frame(x2=control.Pretest[ok],y2=control.Posttest[ok])
  p3 <- ggplot() +
        geom_point(data=frame1,aes(x=x1,y=y1),shape=20) +
        geom_point(data=frame2,aes(x=x2,y=y2),shape=21) +
        scale_y_continuous("Posttest",limits=c(0,125)) +
        scale_x_continuous("Pretest",limits=c(0,125)) +
        theme_bw() +
        labs(title=paste("Grade ",j)) +
        geom_abline(intercept=(beta.mean[1]+beta.mean[2]),slope=beta.mean[3]) +
        geom_abline(intercept=(beta.mean[1]),slope=beta.mean[3],linetype="dashed")
  print(p3, vp = viewport(layout.pos.row = 1, layout.pos.col = j))
}
