stan <- function(model, data, 
                 inits = NULL, iter = 2000, warmup = 1000, thin = 1) {
  
  model <- path.expand(model)
  if(!file.exists(model))  stop(model, "is not a valid file")
  if(!is.data.frame(data)) stop("'data' must be a 'data.frame'")
  if(!is.null(inits)) {
    # check that inits has the correct length and stuff
  }
  msg <- "must be a positive scalar"
  if(length(iter) != 1) stop("'iter'", msg)
  if(iter <= 0) stop("'iter'", msg)
  iter <- as.integer(iter)
  
  if(length(warmup) != 1) stop("'warmup'", msg)
  if(warmup <= 0) stop("'warmup'", msg)
  if(warmup >= iter) stop("'warmup' must be less than 'iter'")
  warmup <- as.integer(warmup)
  
  if(length(thin) != 1) stop("'thin'", msg)
  if(thin <= 0) stop("'thin'", msg)
  if(thin > iter - warmup) stop("'thin' must be smaller than 'iter - warmup'")
  thin <- as.integer(thin)
  
  stanLib <- dirname(system.file(package = "stan"))
  gm <- file.path(stanLib, "inst", "demo", "gm")
  status <- system(paste(".", gm, " ", model, sep = ""))
  if(status != 0) stop("model parsing failed, sorry")

  # compile, link, run
}

