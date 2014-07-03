library(rstan)

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

treatment <- group.id
airport <- scenario.id

n.treatment <- max(treatment)
n.airport <- max(airport)
n <- length(y)

 # flight simulator model
dataList.1 <- list(N=n, n_treatment=n.treatment, n_airport=n.airport, treatment=treatment,
                   airport=airport, y=y)
flight_simulator.sf1 <- stan(file='17.3_flight_simulator.stan', data=dataList.1,
                            iter=1000, chains=4)
print(flight_simulator.sf1)

