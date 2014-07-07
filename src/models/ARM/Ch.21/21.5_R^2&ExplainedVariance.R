## Setting up computations
## be sure to pull out the appropriate error terms (E.B) from a file similar to r_sqr.stan. NOTE THAT THE FILE R_SQR.STAN DOES NOT RUN--IT IS JUST A SIMPLE FRAMEWORK TO FOLLOW.
rsquared <- 1 - mean (apply (e.y, 1, var)) / var (y)
e.a <- E.B[,,1]
e.b <- E.B[,,2]
rsquared.a <- 1 - mean (apply (e.a, 1, var)) / mean (apply (a, 1, var))
rsquared.b <- 1 - mean (apply (e.b, 1, var)) / mean (apply (b, 1, var))
