require(rstan)

mod_def <- read.table(header = TRUE, colClasses = "character",
text = "dir          stancode             datafile
vol1/blocker         blocker.stan         blocker.Rdata 
vol1/bones           bones.stan           bones.Rdata
")

print(mod_def)

STAN_SRC_HOME <- paste0(system.file('include', package = 'rstan'), "/stansrc/") 

BUGS_EX_PATH <- "models/bugs_examples/"

for (i in 1:nrow(mod_def)) {
  scodef <- file.path(STAN_SRC_HOME, BUGS_EX_PATH, mod_def[i, 1], mod_def[i, 2])
  dataf <- file.path(STAN_SRC_HOME, BUGS_EX_PATH, mod_def[i, 1], mod_def[i, 3])
  print(mod_def[i, 1]) 
  modelname <- basename(mod_def[i, 1]) 
  cat("model:", modelname, "\n")

  fit <- stan(file = scodef, data = rstan:::read_rdump(dataf), n_chains = 3)
  print(fit)
  cat("\n\n")
} 


