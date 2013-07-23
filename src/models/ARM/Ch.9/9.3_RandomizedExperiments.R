library(rstan)
library(ggplot2)
library(gridBase)
electric <- read.table ("electric.dat", header=T)
attach(electric)
## Plot of the raw data (Figure 9.4)
  
  electric$Grade <- factor(electric$Grade, levels=c('1','2','3','4'), labels=c('Grade 1', 'Grade 2', 'Grade 3', 'Grade 4'))
  frame = data.frame(x1=control.Posttest,Grade=electric$Grade)
  m1 <- ggplot(frame,aes(x1)) +  geom_histogram(colour = "black", fill = "white", binwidth=5) + theme_bw() + facet_grid(Grade ~.) + theme(strip.text.y = element_blank(),axis.title.y = element_blank(),axis.title.x=element_blank(),strip.background = element_blank()) + labs(title='Test Scores in Control Classes') 

  frame2 = data.frame(x1=treated.Posttest,Grade=electric$Grade)
  m2 <- ggplot(frame2,aes(x1)) +  geom_histogram(colour = "black", fill = "white", binwidth=5) + theme_bw() + facet_grid(Grade ~.) + theme(axis.title.y = element_blank(),axis.ticks.y = element_blank(),axis.text.y=element_blank(),axis.title.x=element_blank()) + labs(title='Test Scores in Treated Classes') 

pushViewport(viewport(layout = grid.layout(1, 2)))
print(m1, vp = viewport(layout.pos.row = 1, layout.pos.col = 1))
print(m2, vp = viewport(layout.pos.row = 1, layout.pos.col = 2))

## Basic analysis of a completely randomized experiment
electric <- read.table ("electric.dat", header=T)
attach(electric)

post.test <- c (treated.Posttest, control.Posttest)
pre.test <- c (treated.Pretest, control.Pretest)
grade <- rep (electric$Grade, 2)
treatment <- rep (c(1,0), rep(length(treated.Posttest),2))
n <- length (post.test)

if (!file.exists("electric_one_pred.sm.RData")) {
    rt <- stanc("electric_one_pred.stan", model_name="electric_one_pred")
    electric_one_pred.sm <- stan_model(stanc_ret=rt)
    save(electric_one_pred.sm, file="electric_one_pred.sm.RData")
} else {
    load("electric_one_pred.sm.RData", verbose=TRUE)
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

## Plot of the regression results (Figure 9.5) FIXME:CONDENSE TO ONE LOOP
if (!file.exists("electric_multi_preds.sm.RData")) {
    rt <- stanc("electric_multi_preds.stan", model_name="electric_multi_preds")
    electric_multi_preds.sm <- stan_model(stanc_ret=rt)
    save(electric_multi_preds.sm, file="electric_multi_preds.sm.RData")
} else {
    load("electric_multi_preds.sm.RData", verbose=TRUE)
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

 # function to make a graph out of the regression coeffs and se's
regression.2tables <- function (name, est1, est2, se1, se2, label1, label2,
                                file, bottom=FALSE){
  J <- length(name)
  name.range <- .6
  x.range <- range (est1+2*se1, est1-2*se1, est2+2*se2, est1-2*se2)
  A <- -x.range[1]/(x.range[2]-x.range[1])
  B <- 1/(x.range[2]-x.range[1])
  height <- .6*J
  width <- 8*(name.range+1)
  gap <- .4
  par (mar=c(4,0,0,0))
  plot (c(-name.range,2+gap), c(3,-J-2), bty="n", xlab="", ylab="", xaxt="n",
     yaxt="n", xaxs="i", yaxs="i", type="n")
  text (-name.range, 2, "Subpopulation", adj=0, cex=.9)
  text (.5, 2, label1, adj=.5, cex=.9)
  text (1+gap+.5, 2, label2, adj=.5, cex=.9)
  lines (c(0,1), c(0,0))
  lines (1+gap+c(0,1), c(0,0))
  lines (c(A,A), c(0,-J-1), lty=2, lwd=.5)
  lines (1+gap+c(A,A), c(0,-J-1), lty=2, lwd=.5)
  ax <- pretty (x.range)
  ax <- ax[(A+B*ax)>0 & (A+B*ax)<1]
  segments (A + B*ax, -.1, A + B*ax, .1, lwd=.5)
  segments (1+gap+A + B*ax, -.1, 1+gap+A + B*ax, .1, lwd=.5)
  text (A + B*ax, .7, ax, cex=.9)
  text (1+gap+A + B*ax, .7, ax, cex=.9)
  text (-name.range, -(1:J), name, adj=0, cex=.9)
  points (A + B*est1, -(1:J), pch=20, cex=.9)
  points (1+gap+A + B*est2, -(1:J), pch=20, cex=.9)
  segments (A + B*(est1-se1), -(1:J), A + B*(est1+se1), -(1:J), lwd=3)
  segments (1+gap+A + B*(est2-se2), -(1:J), 1+gap+A + B*(est2+se2), -(1:J),
            lwd=3)
  segments (A + B*(est1-2*se1), -(1:J), A + B*(est1+2*se1), -(1:J), lwd=.5)
  segments (1+gap+A + B*(est2-2*se2), -(1:J), 1+gap+A + B*(est2+2*se2), -(1:J),
            lwd=.5)
  if (bottom){
    lines (c(0,1), c(-J-1,-J-1))
    lines (1+gap+c(0,1), c(-J-1,-J-1))
    segments (A + B*ax, -J-1-.1, A + B*ax, -J-1+.1, lwd=.5)
    segments (1+gap+A + B*ax, -J-1-.1, 1+gap+A + B*ax, -J-1+.1, lwd=.5)
    text (A + B*ax, -J-1-.7, ax, cex=.9)
    text (1+gap+A + B*ax, -J-1-.7, ax, cex=.9)
  } 
}

#graphs on Figure 9.5
par (mfrow=c(1,1))
 regression.2tables (paste ("Grade", 1:4), est1, est2, se1, se2, 
  "Regression on treatment indicator", "Regression on treatment indicator,
  \ncontrolling for pre-test")

## Controlling for pre-treatment predictors (Figure 9.6)
for (j in 1:4){
  ok <- Grade==j
  pret <- c (treated.Pretest[ok], control.Pretest[ok])
  postt <-c (treated.Posttest[ok], control.Posttest[ok])
  t <- rep (c(1,0), rep(sum(ok),2))
  dataList.1 <- list(N=length(pret), post_test=postt, pre_test=pret,treatment=t)
  electric_multi_preds.sf <- sampling(electric_multi_preds.sm, dataList.1)
  beta.post <- extract(electric_multi_preds.sf, "beta")$beta
  beta.mean <- colMeans(beta.post)

  frame1 = data.frame(x1=treated.Pretest[ok],y1=treated.Posttest[ok])
  frame2 = data.frame(x2=control.Pretest[ok],y2=control.Posttest[ok])
  mm <- ggplot()
  mm <- mm + geom_point(data=frame1,aes(x=x1,y=y1),shape=20)
  mm <- mm + geom_point(data=frame2,aes(x=x2,y=y2),shape=21)
  mm <- mm + scale_y_continuous("Posttest",limits=c(0,125)) + scale_x_continuous("Pretest",limits=c(0,125)) + theme_bw() + labs(title=paste("Grade ",j))
  mm <- mm + geom_abline(intercept=(beta.mean[1]+beta.mean[2]),slope=beta.mean[3])
  paste("mm",j) <- mm + geom_abline(intercept=(beta.mean[1]),slope=beta.mean[3],linetype="dashed")
}
pushViewport(viewport(layout = grid.layout(1, 4)))
print(m[1], vp = viewport(layout.pos.row = 1, layout.pos.col = 1))
print(m[2], vp = viewport(layout.pos.row = 1, layout.pos.col = 2))
print(m[3], vp = viewport(layout.pos.row = 1, layout.pos.col = 3))
print(m[4], vp = viewport(layout.pos.row = 1, layout.pos.col = 4))
