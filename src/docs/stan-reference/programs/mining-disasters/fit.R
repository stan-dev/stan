source("data.R");
fit <- stan("changepoint.stan", data=c("r_e", "r_l", "T", "D"));
fit_ss <- extract(fit);
log_Pr_s <- rep(0,T);
for (t in 1:T)
  log_Pr_s[t] <- mean(fit_ss$lp[,t]);
print(log_Pr_s);

log <- softmax <- function(x) {
  z <- exp(x - max(x));
  return(log(z / sum(z)));
}


library("ggplot2");
qplot(1:T,log(log_softmax(log_Pr_s))) +
 xlab("year") + ylab("log p(change at year)") +
 scale_x_discrete(breaks=c(1875,1900,1925,1950)-1850,
                  labels=c("1875","1900","1925","1950"))

ss_s <- fit_ss$s
earliest <- min(ss_s);
latest <- max(ss_s);
frequency <- rep(0,latest - earliest + 1);
for (n in 1:length(ss_s)) {
  idx <- ss_s[n] - earliest + 1;
  frequency[idx] <- frequency[idx] + 1;
}
year <- 1850 + (earliest:latest);

ggplot(data = data.frame(year=year, frequency=frequency),
       aes(x=year,y=frequency)) +
 geom_bar(stat="identity", fill="white",color="black") +
  xlab("year") + ylab("frequency in 4000 draws")
