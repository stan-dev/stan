stan_demo <- function(model = NULL, ...) {
  # demo examples in src/models, originally written by BG 
  # 
  MODELS_HOME <- file.path(system.file('include', package = 'rstan'), 
                           "stansrc", "models")
  MODELS <- dir(MODELS_HOME, pattern = paste0(model, ".stan", "$"), 
                recursive = TRUE, full.names = TRUE)
  if (length(MODELS) == 0) {
    stop("'model' not found; leave 'model' unspecified to see all choices")
  } else if (length(MODELS) > 1) {
    MODELS <- select.list(MODELS)
    if (MODELS == '') stop("no model was selected") 
    model <- sub(".stan$", "", basename(MODELS))
  }
  MODEL_HOME <- dirname(MODELS)
  STAN_ENV <- new.env()
  # in fact, so far (Wed Oct 10 18:18:58 EDT 2012), only 
  # speed/logistic has logistic_generate_data.Ri, but it 
  # does not create a logistic.Rdata file. So it is 
  # problematic for the logistic example.  
  sim_data_src <- file.path(MODEL_HOME, paste0(model, "_generate_data.R"))
  sim_data_flag <- FALSE 
  tmpdir <- tempdir() 
  if (file.exists(sim_data_src)) { 
    pwd <- getwd()
    on.exit(setwd(pwd)) 
    setwd(tmpdir) 
    source(sim_data_src, local = STAN_ENV, verbose = FALSE)
    sim_data_flag <- TRUE 
    setwd(pwd) 
  } 
  data_file <- if (sim_data_flag) file.path(tmpdir, paste0(model, '.Rdata')) 
               else file.path(MODEL_HOME, paste0(model, '.Rdata')) 
  if (file.exists(data_file)) 
    source(data_file, local = STAN_ENV, verbose = FALSE)
  stan(MODELS, model_name = model, data = STAN_ENV, ...)
}
