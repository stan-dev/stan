stan_demo <- function(model = character(0), ...) {
  if(is.numeric(model)) {
    MODEL_NUM <- model
    model <- character(0)
  }
  else MODEL_NUM <- -1
  MODELS_HOME <- file.path(system.file('include', package = 'rstan'), 
                           "stansrc", "models")
  MODELS <- dir(MODELS_HOME, pattern = paste0(model, ".stan", "$"), 
                recursive = TRUE, full.names = TRUE)
  if(length(MODELS) == 0) {
    stop("'model' not found; leave 'model' unspecified to see all choices")
  }
  else if(length(MODELS) > 1) {
    if(MODEL_NUM %in%  1:length(MODELS)) {
      MODELS <- MODELS[MODEL_NUM]
    }
    else if(MODEL_NUM == 0) MODELS <- ""
    else MODELS <- select.list(MODELS)
    if(!nzchar(MODELS)) {
      return(dir(MODELS_HOME, pattern = paste0(model, ".stan", "$"), 
                 recursive = TRUE, full.names = TRUE))
    }
    model <- sub(".stan$", "", basename(MODELS))
  }
  MODEL_HOME <- dirname(MODELS)
  STAN_ENV <- new.env()
  if(file.exists(gdr <- file.path(MODEL_HOME, paste0(model, "_generate_data.R")))) {
    dir.create(MODEL_HOME <- file.path(tempdir(), model), showWarnings = FALSE)
    stopifnot(file.copy(from = gdr, to = MODEL_HOME, overwrite = TRUE))
    current_wd <- getwd()
    on.exit(setwd(current_wd))
    setwd(MODEL_HOME)
    source(file.path(MODEL_HOME, paste0(model, "_generate_data.R")), 
           local = STAN_ENV, verbose = FALSE)
    }
  if(file.exists(file.path(MODEL_HOME, paste0(model, ".Rdata")))) {
    source(file.path(MODEL_HOME, paste0(model, ".Rdata")), 
           local = STAN_ENV, verbose = FALSE)
  }
  else if(!is.na(fp <- dir(MODEL_HOME, "Rdata$", full.names = TRUE)[1])) {
    source(file.path(fp), local = STAN_ENV, verbose = FALSE)
  }
  return(stan(MODELS, model_name = model, data = STAN_ENV, ...))
}
