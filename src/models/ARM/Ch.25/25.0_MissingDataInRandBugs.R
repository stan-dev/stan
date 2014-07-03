## Read the the Social Indicators Survey data 
# Data are at http://www.stat.columbia.edu/~gelman/arm/examples/sis

wave3 <- read.table ("siswave3v4impute3.csv", header=T, sep=",")
attach(wave3)
n <- nrow (wave3)

# earnings variables:

# rearn:  respondent's earnings
# tearn:  spouse's earnings

# set up some simplified variables to work with
na.fix <- function (a) {
  ifelse (a<0 | a==999999, NA, a)
}

earnings <- na.fix(rearn) + na.fix(tearn)
earnings <- earnings/1000

cbind (sex, race, educ_r, r_age, earnings, police)[91:95,]
