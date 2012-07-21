## take a look at the samples and compare with results computed 
## in other program. 

library(coda) 
post <- read.csv(file = "samples.csv", header = TRUE, comment.char = '#'); 

# copied from JAGS classic-bugs example 
"benchstats" <-
structure(c(-4.7831624, 0.00291866339160000, -1.968915345, 0.265791517214, 
0.75342247, -1.0063284139, 1.245859934, 0.7179099125, 0.991958536862955, 
0.0167522354569816, 0.545071164414747, 0.613006982741994, 0.612254543123539, 
0.834773133087595, 0.179659976299101, 0.362347551415367, 0.0313684832093504, 
0.000529752199434935, 0.0172366636643072, 0.0193849828705223, 
0.0193611886405616, 0.0263978443007167, 0.00568134729477027, 
0.0114584356705753, 0.0521436954261121, 0.000801389725300001, 
0.0215086114888169, 0.0260244994849658, 0.0268674571467801, 0.0331471966526037, 
0.00953117547520882, 0.0186457130325326), .Dim = as.integer(c(8, 
4)), .Dimnames = list(c("alpha", "beta.age", "beta.sex", "beta.disease[2]", 
"beta.disease[3]", "beta.disease[4]", "r", "sigma"), c("Mean", 
"SD", "Naive SE", "Time-series SE")))


poi.names <- c("alpha", "beta_age", "beta_sex", 
               "beta_disease2", "beta_disease3", "beta_disease4", 
               "r", "sigma"); 
poi <- post[, poi.names] 
poi <- as.mcmc(poi); 
summary(poi) 

print(benchstats) 

# [Wed Dec  7 19:55:27 EST 2011] 
# the results are closer to those presented by WinBUGS than to
# the JAGS benchstats 

