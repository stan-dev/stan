## R script to plot performance.csv
## It should be run from the folder that contains performance.csv,
## typically test/performance/

performance <- read.csv("performance.csv")

runs <- performance[,7:106]

x <- 1:nrow(performance)
xlim <- c(1, nrow(performance)) + c(-0.5, 0.5)
ylim <- c(0, ceiling(max(runs)))

performance$mean <- apply(runs, 1, mean)
performance$min <- apply(runs, 1, min)
performance$max <- apply(runs, 1, max)
performance$lo_25 <- apply(runs, 1, quantile, 0.25)
performance$hi_75 <- apply(runs, 1, quantile, 0.75)

label <- paste(substr(performance$git.hash, 1, 7),
               " ",
               format(as.Date(performance$git.date, format="%a %b %d %H:%M:%S %Y"),
                      "%m-%d-%Y"))
col_index <- ifelse(performance$matches.tagged.version == "yes" &
                    performance$all.values.same == "yes", 1, 2)

cols <- rbind(c("gray", "black"), c("red4", "red"))

png("performance.png", 900, 550)
par(mar = c(6, 4, 2, 0.5))
plot(NA,
     xlim=xlim, ylim=ylim,
     xaxs="i", yaxs="i",
     bty="l",
     main="logistic regression",
     xlab="", ylab="time (s)",
     type="n", xaxt="n")
axis(1, at=x,
     labels=FALSE)
text(x=x, y=-0.150, labels = label, srt=60,
     xpd=TRUE, cex=0.5, adj=1)

points(x, performance$mean, col=cols[col_index, 2])
for (n in 1:nrow(performance)) {
  segments(x[n], performance$min[n], x[n], performance$max[n], col=cols[col_index[n], 1])
  segments(x[n], performance$lo_25[n], x[n], performance$hi_75[n], col=cols[col_index[n], 2], lwd=2)
  segments(x[n]-0.2, performance$lo_25[n], x[n]+0.2, performance$lo_25[n], col=cols[col_index[n], 2], lwd=2)
  segments(x[n]-0.2, performance$hi_75[n], x[n]+0.2, performance$hi_75[n], col=cols[col_index[n], 2], lwd=2)
}
dev.off()
