library(rstan)
library(ggplot2)
source("9.3_Randomized experiments.R") # where data was cleaned

## Observational studies

 # Plot Figure 9.9

supp <- c(as.numeric(electric[,"Supplement."])-1, rep(NA,nrow(electric)))
# supp=0 for replace, 1 for supplement, NA for control

est1 <- rep(NA,4)
se1 <- rep(NA,4)
for (k in 1:4){
  ok <- (grade==k) & (!is.na(supp))
  lm.supp <- lm (post.test ~ supp + pre.test, subset=ok)
  est1[k] <- lm.supp$coef[2]
  se1[k] <- summary(lm.supp)$coef[2,2]
}

 # function to make a graph out of the regression coeffs and se's

regression.2tablesA <- function (name, est1, se1, label1, file, bottom=FALSE){
  J <- length(name)
  name.range <- .6
  x.range <- range (est1+2*se1, est1-2*se1)
  A <- -x.range[1]/(x.range[2]-x.range[1])
  B <- 1/(x.range[2]-x.range[1])
  height <- .6*J
  width <- 8*(name.range+1)
  gap <- .4
  
  par (mar=c(4,0,0,0))
  plot (c(-name.range,2+gap), c(3,-J-2), bty="n", xlab="", ylab="",
        xaxt="n", yaxt="n", xaxs="i", yaxs="i", type="n")
  text (-name.range, 2, "Subpopulation", adj=0, cex=.9)
  text (.5, 2, label1, adj=.5, cex=.9)
  lines (c(0,1), c(0,0))
  lines (c(A,A), c(0,-J-1), lty=2, lwd=.5)
  ax <- pretty (x.range)
  ax <- ax[(A+B*ax)>0 & (A+B*ax)<1]
  segments (A + B*ax, -.1, A + B*ax, .1, lwd=.5)
  text (A + B*ax, .7, ax, cex=.9)
  text (-name.range, -(1:J), name, adj=0, cex=.9)
  points (A + B*est1, -(1:J), pch=20, cex=1)
  segments (A + B*(est1-se1), -(1:J), A + B*(est1+se1), -(1:J), lwd=3)
  segments (A + B*(est1-2*se1), -(1:J), A + B*(est1+2*se1), -(1:J), lwd=.5)
  if (bottom){
    lines (c(0,1), c(-J-1,-J-1))
    segments (A + B*ax, -J-1-.1, A + B*ax, -J-1+.1, lwd=.5)
    text (A + B*ax, -J-1-.7, ax, cex=.9)
  } 
}
                  # graphs on Figure 9.9

regression.2tablesA (paste ("Grade", 1:4), est1, se1, "Estimated effect of supplement,
    \ncompared to replacement")

## Examining overlap in the Electric Company example (Figure 9.12)

par (mfrow=c(2,2), mar=c(5,4,2,2))
for (j in 1:4){
  ok <- (grade==j) & (!is.na(supp))
  plot(pre.test[ok],post.test[ok], xlab="pre-test score", 
    ylab="posttest score", pch=20, type="n", main=paste("grade",j))
  fit.1 <- lm (post.test ~ pre.test + supp, subset=ok)
  abline (fit.1$coef[1], fit.1$coef[2], lwd=.5, col="gray")
  abline (fit.1$coef[1]+ fit.1$coef[3], fit.1$coef[2], lwd=.5)
  points (pre.test[supp==1 & ok], post.test[supp==1 & ok], pch=20, 
    cex=1.1)
  points (pre.test[supp==0 & ok], post.test[supp==0 & ok], pch=20,
    cex=1.1, col="gray")
}
