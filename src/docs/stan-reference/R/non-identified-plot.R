library(ggplot2);
library(grid);

N <- 100;
y <- rnorm(N);

loc_sum <- function(lambda1,lambda2) {
  sum(dnorm(y,lambda1 + lambda2, 1, log=TRUE));
}

loc_sum_prior <- function(lambda1,lambda2) {
  sum(dnorm(y,lambda1 + lambda2, 1, log=TRUE)) + 
  sum(dnorm(c(lambda1,lambda2),0,1,log=TRUE));
}

one_param <- function(xs) {  
  dta <- rnorm(100,0,1);
  result <- rep(NA, length(xs));
  i <- 1;
  for (x in xs) {
    result[i] <- sum(dnorm(dta,xs[i],1,log=TRUE));
    i <- i + 1;
  } 
  return(result);
}
p_one_param <-
  ggplot(data.frame(mu=c(-20, 20)), aes(mu)) +
  labs(title = "Proper Posterior (without Prior)\n") + 
  stat_function(fun=one_param) +
  labs(x=expression(mu), y="log p") +
  theme(aspect.ratio=1,
        panel.border=element_blank(),
        plot.margin=unit(c(0,0,0,0),"cm"),
        text=element_text(size=28),
        axis.title=element_text(size=32));
png(res=100,height=800,width=900,file="one_param_identified.png");
print(p_one_param);
dev.off();





K <- 500;
ub <- 25;
lambda_1 <- seq(-ub,ub,len=K);
lambda_2 <- seq(-ub,ub,len=K);

v_lambda_1 <- rep(NA,K^2);
v_lambda_2 <- rep(NA,K^2);
v_density <- rep(NA,K^2);

# use prior (loc_sum_prior)
pos <- 1;
for (m in 1:K) {
  for (n in 1:K) {
    v_lambda_1[pos] <- lambda_1[m];
    v_lambda_2[pos] <- lambda_2[n];
    v_density[pos] <- loc_sum_prior(lambda_1[m], lambda_2[n]);
    pos <- pos + 1;
  }
}
df_vals <- list(lambda_1=v_lambda_1,lambda_2=v_lambda_2,log_density=v_density);
df_prior <- as.data.frame(df_vals);

# plot and save as png

p_prior <- 
 ggplot(df_prior,aes(x=lambda_1,y=lambda_2,fill=log_density)) +
 labs(title = "Proper Posterior (with Prior)\n") + 
 geom_tile() +
 labs(x=expression(lambda[1]), y=expression(lambda[2])) + 
 scale_fill_gradient2("log p",
                      limits=c(-600,-120),
                      midpoint=-400, low="gray95", high="darkblue",
                      mid="lightyellow", na.value="transparent",      
                      breaks=c(-200, -400, -600),   
                      labels=c("-200", "-400", "-600")) +
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_continuous(expand = c(0, 0)) +
  theme(aspect.ratio=1,
        panel.border=element_blank(),
        plot.margin=unit(c(0,0,0,0),"cm"),
        text=element_text(size=28),
        axis.title=element_text(size=32),
        legend.title=element_text(size=30,face="plain",color="gray25"), 
        legend.text=element_text(size=20,color="gray40"),
        legend.margin=unit(0.3,"cm") )
  
png(res=100,height=800,width=900,file="non-identified-plus-prior.png");
plot(p_prior)
dev.off();
plot(p_prior)

# no prior (loc_sum)
pos <- 1;
for (m in 1:K) {
  for (n in 1:K) {
    v_density[pos] <- loc_sum(lambda_1[m], lambda_2[n]);
    pos <- pos + 1;
  }
}
df_vals <- list(lambda_1=v_lambda_1,lambda_2=v_lambda_2,log_density=v_density);
df_no_prior <- as.data.frame(df_vals);

# plot and save as png

p_no_prior <- 
 ggplot(df_no_prior,aes(x=lambda_1,y=lambda_2,fill=log_density)) +
 labs(title = "Improper Posterior (without Prior)\n") + 
 geom_tile() +
 labs(x=expression(lambda[1]), y=expression(lambda[2])) +
 scale_x_continuous(expand = c(0, 0)) +
 scale_y_continuous(expand = c(0, 0)) +
 scale_fill_gradient2("log p",
                      limits=c(-600,-120),
                      midpoint=-400, low="gray95", high="darkblue",
                      mid="lightyellow", na.value="transparent",
                      breaks=c(-200, -400, -600),   
                      labels=c("-200", "-400", "-600") ) +
 theme(aspect.ratio=1,
        panel.border=element_blank(),
        plot.margin=unit(c(0,0,0,0),"cm"),
        text=element_text(size=28),
        axis.title=element_text(size=32),
        legend.title=element_text(size=30,face="plain",color="gray25"), 
        legend.text=element_text(size=20,color="gray40"),
        legend.margin=unit(0.3,"cm") )


  theme(aspect.ratio=1,text=element_text(size=28),
        axis.title=element_text(size=32),
        legend.margin=unit(0.3,"cm"));

png(res=100,height=800,width=900,file="non-identified.png")
print(p_no_prior)
dev.off();
