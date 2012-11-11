library("ggplot2");

p_funnel <- function(x1,y) {
  return(dnorm(y,0.0,3.0,log=TRUE) + dnorm(x1,0.0,exp(y/2),log=TRUE));
}

K = 200;

x1 <- rep(0,(K+1)^2);
y <- rep(0,(K+1)^2);
log_p_y_x1 <- rep(0,(K+1)^2);
pos <- 1;
for (m in 1:(K+1)) {
  for (n in 1:(K+1)) {
    y[pos] <- -9.0 + 18.0 * (m - 1) / K;    
    x1[pos] <- -20.0 + 40.0 * (n - 1) / K;
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
     labs(title = "Funnel Density (log scale)\n") +
     geom_tile() +
     scale_x_continuous("x[1]",expand=c(0,0),limits=c(-20,20), breaks=c(-20,-10,0,10,20)) +
     scale_y_continuous(expand=c(0,0),limits=c(-9,9), breaks=c(-9,-6,-3,0,3,6,9)) +
     scale_fill_gradient2("log p(y,x[1])\n",
                          limits=c(-18,2),   
                          midpoint=-6.75, mid="lightyellow",
                          low="gray95", high="darkblue", na.value="transparent",
                          breaks=c(0,-8,-16),
                          labels=c("0","-8","-16"));

png(filename="funnel.png", width=1500,height=1200,res=300);
print(funnel_plot);
dev.off();                    

