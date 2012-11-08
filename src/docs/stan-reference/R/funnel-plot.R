library("ggplot2");

p_funnel <- function(x1,y) {
  return(dnorm(y,0.0,3.0,log=TRUE) + dnorm(x1,0.0,exp(y),log=TRUE));
}

K = 500;

x1 <- rep(0,(K+1)^2);
y <- rep(0,(K+1)^2);
log_p_y_x1 <- rep(0,(K+1)^2);
pos <- 1;
for (m in 1:(K+1)) {
  for (n in 1:(K+1)) {
    y[pos] <- -6.0 + 12.0 * (m - 1) / K;    
    x1[pos] <- -30.0 + 60.0 * (n - 1) / K;
    log_p_y_x1[pos] <- p_funnel(x1[pos],y[pos]);
    pos <- pos + 1;
  }
}

library("scales");

df <- data.frame(x1=x1,
                y=y,
                log_p_y_x1=log_p_y_x1);
funnel_plot <-
     ggplot(df, aes(x1,y,fill = log_p_y_x1)) +
     labs(title = "Funnel Density\n") +
     geom_tile() +
     scale_x_continuous("x[1]",expand=c(0,0),limits=c(-30,30)) +
     scale_y_continuous(expand=c(0,0),limits=c(-4,4), breaks=c(-4,-2,0,2,4)) +
     scale_fill_gradient2("log p(y,x[1])\n",
                          limits=c(-8,1),midpoint=-3,
                          low="white", mid="black", high="black", na.value="white");                       
#                         midpoint=-7, 
#                          low="white", mid="gray", high="black", na.value="white");
#                          low="orange", mid="white", high="blue", na.value="white");
#                          breaks=c(0,-25,-50,-75,-100),
#                          labels=c("0","-25","-50","-75","< -100"));

png(filename="funnel.png", width=1500,height=1200,res=300);
print(funnel_plot);
dev.off();                    

