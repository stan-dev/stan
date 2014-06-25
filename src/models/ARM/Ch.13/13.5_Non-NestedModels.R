library(rstan)
library(ggplot2)

## Read the pilots data & define variables
# Data are at http://www.stat.columbia.edu/~gelman/arm/examples/pilots

pilots <- read.table ("pilots.dat", header=TRUE)
attach (pilots)
group.names <- as.vector(unique(group))
scenario.names <- as.vector(unique(scenario))
n.group <- length(group.names)
n.scenario <- length(scenario.names)
successes <- NULL
failures <- NULL
group.id <- NULL
scenario.id <- NULL
for (j in 1:n.group){
  for (k in 1:n.scenario){
    ok <- group==group.names[j] & scenario==scenario.names[k]    
    successes <- c (successes, sum(recovered[ok]==1,na.rm=T))
    failures <- c (failures, sum(recovered[ok]==0,na.rm=T))
    group.id <- c (group.id, j)
    scenario.id <- c (scenario.id, k)
  }
}

y <- successes/(successes+failures)
y.mat <- matrix (y, n.scenario, n.group)
sort.group <- order(apply(y.mat,2,mean))
sort.scenario <- order(apply(y.mat,1,mean))

group.id.new <- sort.group[group.id]
scenario.id.new <- sort.scenario[scenario.id]
y.mat.new <- y.mat[sort.scenario,sort.group]

scenario.abbr <- c("Nagoya", "B'ham", "Detroit", "Ptsbgh", "Roseln", "Chrlt", "Shemya", "Toledo")

## Model fit
## M1 <- lmer (y ~ 1 + (1 | group.id) + (1 | scenario.id))
dataList.1 <- list(N=length(y),y=y,n_groups=n.group,n_scenarios=n.scenario,group_id=group.id,scenario_id=scenario.id)
pilots.sf1 <- stan(file='pilots.stan', data=dataList.1, iter=20000, chains=4)
print(pilots.sf1,pars = c("a","b", "sigma_y", "lp__"))

## Plot figure 13.8

image (y.mat.new, col=gray((1:11)/12), xaxt="n", yaxt="n")
axis (2, seq(0,1,length=n.group), group.names[sort.group], tck=0, cex.axis=1.2)
axis (3, seq(0,1,length=n.scenario), scenario.abbr[sort.scenario], tck=0, cex.axis=1.2)
for (x in seq(.5,n.group-.5,1)/(n.group-1)) lines (c(-10,10),rep(x,2),col="white", lwd=.5)
for (x in seq(.5,n.scenario-.5,1)/(n.scenario-1)) lines (rep(x,2),c(-10,10),col="white", lwd=.5)

########################################################################################################
## Read the data
# Data are at http://www.stat.columbia.edu/~gelman/arm/examples/earnings

# The R codes & data files should be saved in the same directory for
# the source command to work
library(foreign)
heights <- read.dta ("heights.dta")
attach(heights)

  # define variables 
age <- 90 - yearbn                     # survey was conducted in 1990
age[age<18] <- NA
age.category <- ifelse (age<35, 1, ifelse (age<50, 2, 3))
eth <- ifelse (race==2, 1, ifelse (hisp==1, 2, ifelse (race==1, 3, 4)))
male <- 2 - sex

  # (for simplicity) remove cases with missing data
ok <- !is.na (earn+height+sex+age.category+eth) & earn>0 & yearbn>25
heights.clean <- as.data.frame (cbind (earn, height, sex, race, hisp, ed, age,
    age.category, eth, male)[ok,])
n <- nrow (heights.clean)
height.jitter.add <- runif (n, -.2, .2)

 # rename variables
y <- log(earn[ok])
x <- height[ok]
n <- length(y[ok])
n.age <- 3
n.eth <- 4
age <- age.category
age.ok <- age[ok]
eth.ok <- eth[ok]

## Regression centering the predictors
##M1 <- lmer (y ~ x.centered + (1 + x.centered | eth) + (1 + x.centered | age) + (1 + x.centered | eth:age))
x.centered <- x - mean(x)

dataList.2 <- list(N=length(y),y=y,x=x.centered,n_eth=n.eth,n_age=n.age,eth=eth.ok,age=age[ok])
earnings_latin_square.sf1 <- stan(file='earnings_latin_square.stan',
                                  data=dataList.2, iter=1000, chains=4)
print(earnings_latin_square.sf1,pars = c("a1","a2","b1","b2", "c","d","sigma_y", "lp__"))
post <- extract(earnings_latin_square.sf1)

 # plot figure 13.10
age.label <- c("AGE 18-34", "AGE 35-49", "AGE 50-64")
eth.label <- c("BLACKS","HISPANICS","WHITES","OTHERS")
x <- height[ok] - mean(height[ok])
pushViewport(viewport(layout = grid.layout(3, 4)))
a.hat.M1 <- colMeans(post1$a)
b.hat.M1 <- colMeans(post1$b)

for (j in 1:3) {
  for (k in 1:4) {
    frame1 = data.frame(y1=y[age.ok==j&eth.ok==k],x1=x.centered[age.ok==j&eth.ok==k])
    p1 <- ggplot(frame1, aes(x=x1, y=y1)) +
          geom_jitter(position = position_jitter(width = .05, height = 0)) +
          scale_x_continuous("Height (inches from mean)") +
          scale_y_continuous("log(earnings)") +
          labs(title=paste(eth.label[k],", ",age.label[j])) +
          theme_bw() +
          geom_abline(aes(intercept = colMeans(post$a[,1])[k]
                                       + colMeans(post$b[,1])[j]
                                       + colMeans(post$c[,1])[k*j],
                          slope=colMeans(post$a[,2])[k]
                                 + colMeans(post$b[,2])[j]
                                 + colMeans(post$c[,2])[k*j]), size = 0.25)
    for (s in 1:20)
      p1 <- p1 + geom_abline(intercept=(post$a[4000-s,1])[k]
                             + (post$b[4000-s,1])[j] + (post$c[4000-s,1])[k*j],
                              slope=(post$a[4000-s,2])[k]
                              + (post$b[4000-s,2])[j]
                              + (post$c[4000-s,2])[k*j],colour="grey10")
    print(p1, vp = viewport(layout.pos.row = j, layout.pos.col = k))
  }
}













