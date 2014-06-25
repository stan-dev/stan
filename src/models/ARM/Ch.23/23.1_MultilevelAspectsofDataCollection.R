library(rstan)
library(ggplot2)
library(gridBase)

## Read the data
# Data are at http://www.stat.columbia.edu/~gelman/arm/examples/electric.company

# The R codes & data files should be saved in the same directory for
# the source command to work

electric <- read.table ("electric.dat", header=T)
attach(electric)

post.test <- c (treated.Posttest, control.Posttest)
pre.test <- c (treated.Pretest, control.Pretest)
grade <- rep (electric$Grade, 2)
treatment <- rep (c(1,0), rep(length(treated.Posttest),2))
n <- length (post.test)

## Hierarchical model including pair indicators
pair <- rep (1:nrow(electric), 2)
grade.pair <- grade[1:nrow(electric)]
y <- post.test
n <- length(y)
n.grade <- max(grade)
n.pair <- max(pair)


# fitting all 4 grades at once (without controlling for pretest)

dataList.1 <- list(N=n,y=y,n_grade=n.grade,grade=grade,grade_pair=grade.pair,
                   treatment=treatment,n_pair=n.pair,pair=pair,n_grade_pair=4)
electric_1a.sf1 <- stan(file='electric_1a.stan', data=dataList.1, iter=1000,
                        chains=4)
print(electric_1a.sf1, pars = c("a","beta","lp__"))


# simple model controlling for pre-test

dataList.2 <- list(N=n,y=y,treatment=treatment,n_pair=n.pair,pair=pair,
                   pre_test=pre.test)
electric_1b.sf1 <- stan(file='electric_1b.stan', data=dataList.2,
                        iter=1000, chains=4)
print(electric_1b.sf1, pars = c("a","beta","lp__"))

# fitting all 4 grades at once (controlling for pretest)

dataList.3 <- list(N=n,y=y,n_grade=n.grade,grade=grade,grade_pair=grade.pair,
                   treatment=treatment,n_pair=n.pair,pair=pair,pre_test=pre.test,
                   n_grade_pair=4)
electric_1c.sf1 <- stan(file='electric_1c.stan', data=dataList.3,
                        iter=1000, chains=4)
print(electric_1c.sf1, pars = c("a","beta","beta2","lp__"))


## Plot Figure 23.1
post.1a <- extract(electric_1a.sf1)
post.1c <- extract(electric_1c.sf1)

est3 <- colMeans(post.1a$beta)
se3 <- apply(post.1a$beta,2,sd)
est4 <- colMeans(post.1c$beta)
se4 <- apply(post.1c$beta,2,sd)

frame = data.frame(Grade=4:1,x1=est3,x2=est4,se1=se3,se2=se4)
p1 <- ggplot(frame, aes(x=x1,y=Grade)) +
      geom_point(size=3) +
      theme_bw() +
      labs(title="Regression on Treatment Indicator Controlling for Pairs") +
      geom_segment(aes(x=x1-se1,y=Grade,xend=x1+se1,yend=Grade),size=2) + 
      geom_segment(aes(x=x1-2 * se1,y=Grade,xend=x1+2*se1,yend=Grade)) +
      geom_vline(xintercept=0,linetype="dotted") +
      scale_x_continuous("Effect of the Electric Company TV Show")

p2 <- ggplot(frame, aes(x=x2,y=Grade)) +
      geom_point(size=3) +
      theme_bw() +
      labs(title="Regression on Treatment Indicator Controlling for Pairs and Pre-test") +
      geom_segment(aes(x=x2-se2,y=Grade,xend=x2+se2,yend=Grade),size=2) + 
      geom_segment(aes(x=x2-2 * se2,y=Grade,xend=x2+2*se2,yend=Grade)) +
      geom_vline(xintercept=0,linetype="dotted") +
      scale_x_continuous("Effect of the Electric Company TV Show")

pushViewport(viewport(layout = grid.layout(1, 2)))
print(p1, vp = viewport(layout.pos.row = 1, layout.pos.col = 1))
print(p2, vp = viewport(layout.pos.row = 1, layout.pos.col = 2))

## Figure 23.2 comparing all the se's

 # define est1, est2, se1, se2 (from models on chapter 9)

est1 <- rep(NA,4)
est2 <- rep(NA,4)
se1 <- rep(NA,4)
se2 <- rep(NA,4)

for (k in 1:4) {
post.test1 <- post.test[grade==k]
treatment1 <- treatment[grade==k]
pre.test1 <- pre.test[grade==k]
ok <- !is.na(post.test1+treatment1+pre.test1)
post.test2 <- post.test1[ok]
treatment2 <- treatment1[ok]
pre.test2 <- pre.test1[ok]
N1 <- length(post.test2)

dataList.1 <- list(N=N1, post_test=post.test2, treatment=treatment2)
electric_one_pred.sf1 <- stan(file='electric_one_pred.stan', data=dataList.1,
                              iter=1000, chains=4)
print(electric_one_pred.sf1)
beta.post <- extract(electric_one_pred.sf1, "beta")$beta
est1[k] <- colMeans(beta.post)[2]
se1[k] <- sd(beta.post[,2])

dataList.2 <- list(N=N1, post_test=post.test2, treatment=treatment2,
                   pre_test=pre.test2)
electric_multi_preds.sf1 <- stan(file='electric_multi_preds.stan',
                                 data=dataList.2, iter=1000, chains=4)
print(electric_multi_preds.sf1)
beta.post2 <- extract(electric_one_pred.sf1, "beta")$beta
est2[k] <- colMeans(beta.post2)[2]
se2[k] <- sd(beta.post2[,2])
}

 # make the plot

# analyze replace/supplement
supp <- c(as.numeric(electric[,"Supplement."])-1, rep(NA,nrow(electric)))
# supp=0 for replace, 1 for supplement, NA for control

ses <- cbind(se1,se2,se3,se4)

dev.new()
pushViewport(viewport(layout = grid.layout(1, 4)))
for (j in 1:4){
  se.grade <- ses[j,]
  x <- c(1:4)
  frame1 = data.frame(x1=x,y1=se.grade)
  p2 <- ggplot(frame1, aes(x=x1,y=y1)) +
        geom_point() +
        scale_y_continuous("S.E of Estimated Treatment Effect") +
        theme_bw() +
        labs(title=paste("Grade ",j))
  print(p2, vp = viewport(layout.pos.row = 1, layout.pos.col = j))
}
