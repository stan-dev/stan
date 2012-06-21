require(rstan)

mod.def <- read.table(header = TRUE, colClasses = "character",
text = "dir          stancode             datafile
vol1/blocker         blocker.stan         blocker.Rdata 
vol1/bones           bones.stan           bones.Rdata
")

print(mod.def)

STAN_HOME <- Sys.getenv("STAN_HOME")

BUGS_EX_PATH <- "/src/models/bugs_examples/"

for (i in 1:nrow(mod.def)) {
  scodef <- file.path(STAN_HOME, BUGS_EX_PATH, mod.def[i, 1], mod.def[i, 2])
  dataf <- file.path(STAN_HOME, BUGS_EX_PATH, mod.def[i, 1], mod.def[i, 3])
  print(mod.def[i, 1]) 
  modelname <- basename(mod.def[i, 1]) 
  cat("model:", modelname, "\n")

  fit <- stan(file = scodef, data = rstan:::read.rdump(dataf), n.chains = 3)
  print(fit)
  cat("\n\n")
} 


