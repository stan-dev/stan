library(rstan)
library(ggplot2)
library(gridBase)
source("9.3_Randomized experiments.R") # where data was cleaned

## Observational studies

 # Plot Figure 9.9

supp <- c(as.numeric(electric[,"Supplement."])-1, rep(NA,nrow(electric)))
# supp=0 for replace, 1 for supplement, NA for control

est1 <- rep(NA,4)
se1 <- rep(NA,4)
for (k in 1:4){
  ok <- (grade==k) & (!is.na(supp))
  dataList.1 <- list(N=length(post.test[ok]), post_test=post.test[ok], treatment=supp[ok],pre_test=pre.test[ok])
  electric_multi_preds.sf1 <- sampling(electric_multi_preds.sm, dataList.1)
  print(electric_multi_preds.sf1)
  beta.post <- extract(electric_multi_preds.sf1, "beta")$beta
  est1[k] <- colMeans(beta.post)[2]
  se1[k] <- sd(beta.post[,2])
}

# graphs on Figure 9.9

frame = data.frame(Grade=4:1,x1=est1,se1=se1)
dev.new()
p1 <- ggplot(frame, aes(x=x1,y=Grade)) +
      geom_point(size=3) +
      theme_bw() +
      labs(title="Estimated Effect of Supplement Compared to Replacement") +
      geom_segment(aes(x=x1-se1,y=Grade,xend=x1+se1,yend=Grade),size=2) + 
      geom_segment(aes(x=x1-2 * se1,y=Grade,xend=x1+2*se1,yend=Grade)) +
      geom_vline(xintercept=0,linetype="dotted") +
      scale_x_continuous("")

print(p1)

## Examining overlap in the Electric Company example (Figure 9.12)
dev.new()
pushViewport(viewport(layout = grid.layout(1, 4)))

for (j in 1:4){
  ok <- (grade==j) & (!is.na(supp))
  frame2 = data.frame(x1=pre.test[ok],y1=post.test[ok],x2=supp[ok])
  dataList.2 <- list(N=length(frame2$x1),post_test=frame2$y1,
                     treatment=frame2$x1,pre_test=frame2$x2)
  electric_multi_preds.sf <- sampling(electric_multi_preds.sm, dataList.2)
  beta.post <- extract(electric_multi_preds.sf, "beta")$beta
  beta.mean <- colMeans(beta.post)
  
  pre.test2 <- pre.test[ok&supp==1]
  post.test2 <- post.test[ok&supp==1]
  frame3 = data.frame(x1=pre.test2,y1=post.test2)

  pre.test3 <- pre.test[ok&supp==0]
  post.test3 <- post.test[ok&supp==0]
  frame4 = data.frame(x1=pre.test3,y1=post.test3)

  p3 <- ggplot() +
        geom_point(data=frame3,aes(x=x1,y=y1),shape=20) +
        geom_point(data=frame4,aes(x=x1,y=y1),shape=20,colour="gray") +    
        scale_y_continuous("Posttest") +
        scale_x_continuous("Pretest") +
        theme_bw() +
        labs(title=paste("Grade ",j)) +
        geom_abline(intercept=(beta.mean[1]+beta.mean[3]),slope=beta.mean[2],colour="gray30") +
        geom_abline(intercept=(beta.mean[1]),slope=beta.mean[2])
  print(p3, vp = viewport(layout.pos.row = 1, layout.pos.col = j))
}
