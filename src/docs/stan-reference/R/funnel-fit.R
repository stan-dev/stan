#funnel_fit <- stan(file='../../../models/misc/funnel/funnel_reparam.stan', 
#                   iter=50000, thin=1);
funnel_samples <- extract(funnel_fit,permuted=TRUE,inc_warmup=FALSE);
funnel_df <- data.frame(x1=funnel_samples$x[,1],y=funnel_samples$y)
ggp <- 
  ggplot(funnel_df,aes(x=x1,y=y)) +
  coord_cartesian(xlim=c(-30,30), ylim=c(-4,4)) + 
  scale_x_continuous("x[1]", expand=c(0,0)) + 
  scale_y_continuous("y", expand=c(0,0), breaks=c(-4,-2,0,2,4)) +
  labs(title="Transformed Funnel Samples\n") +
  geom_point(shape='.', alpha=1/15)


png(filename="funnel-fit.png", width=1200,height=1200,res=300);
print(ggp);
dev.off();                    
