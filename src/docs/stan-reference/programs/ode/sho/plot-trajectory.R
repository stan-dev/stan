# solutions from running sho-sim, with added (0,1) init
library(plyr);
library(ggplot2);
y <- structure(c(1,0.397095,-0.300526,-0.948297,-0.57433,-0.00316177,
                 0.675424,0.703672,-0.0918871,-0.469737,-0.429309,
                 0,-0.853654,-0.696679,0.0545529,0.634292,0.772725,
                 0.15645,-0.424857,-0.487648,-0.298427,0.16981),
               .Dim=c(11,2))
ydf <- data.frame(y);
ydf <- cbind(ydf, 0:10);
ydf <- rename(ydf, c("X1"="y1","X2"="y2","0:10"="t"))

# won't run from script; need to cut-and-paste this in
pdf(file="sho-trajectory.pdf",width=4,height=4)
qplot(y1,y2, data=ydf, geom=c("point","path")) +
  annotate("text", x=1, y=0.05, label="t=0")
dev.off()
