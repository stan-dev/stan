funnel_fit <- stan(file='../../../models/misc/funnel/funnel_reparam.stan')
funnel_samples <- extract(funnel_fit,permuted=TRUE,inc_warmup=FALSE);
funnel_df <- data.frame(x1=funnel_samples$x[,1],y=funnel_samples$y[])
ggp <- 
  ggplot(funnel_df,aes(x=x1,y=y)) +
  coord_cartesian(xlim=c(-20,20), ylim=c(-9,9)) + 
  scale_x_continuous("x[1]", expand=c(0,0), breaks=c(-20,-10,0,10,20)) + 
  scale_y_continuous("y", expand=c(0,0), breaks=c(-9,-6,-3,0,3,6,9)) +
  labs(title="Funnel Samples (transformed model)\n") +
  geom_point(shape='.', alpha=1, color="black")


png(filename="funnel-fit.png", width=1200,height=1200,res=300);
print(ggp);
dev.off();                    
