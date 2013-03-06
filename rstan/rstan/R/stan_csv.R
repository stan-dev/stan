get_sampler_name <- function(leapfrog_steps, equal_step_sizes, nondiag_mass) {
  if (!is.null(nondiag_mass) && nondiag_mass > 0) 
    return("NUTS(nondiag)")
  if (leapfrog_steps < 0) {
    if (equal_step_sizes == 0) return("NUTS2")
    return("NUTS1")
  } 
  return("HMC")
} 

paridx_fun <- function(names) {
  # Args:
  #   names: names (character vector) such as lp__, treedepth__, stepsize__,
  #          alpha, beta.1, 
  # Returns: 
  #   The indexes in the names that are parameters other than lp__,
  #   treedepth__, or stepsize__. The vector has atttibute meta
  #   with the indexes of 'treedepth__', 'lp__', and 'stepsize__'
  #   if available. 
  
  metaidx <- match(c('lp__', 'treedepth__', 'stepsize__'), names)
  names(metaidx) <- c('lp__', 'treedepth__', 'stepsize__')
  paridx <- setdiff(seq_along(names), metaidx)
  attr(paridx, "meta") <- metaidx[!sapply(metaidx, is.na)]
  paridx
}

parse_stancsv_comments <- function(comments) {
  # Parse the comments in Stan CSV files to get information such as
  # iter, thin, seed, etc. This is specific to the CSV files
  # generated from Stan

  nuts_diag_lineno <- which(grepl('(mcmc::nuts_diag)', comments))
  nuts_nondiag_lineno <- which(grepl('(mcmc::nuts_nondiag)', comments))
  adaptation_info <- character(0)  
  len <- length(comments)
  nondiag_mass <- 0
  if (length(nuts_diag_lineno) > 0) {  
    adaptation_info <- paste(comments[nuts_diag_lineno:len], collapse = '\n')
    comments <- comments[1:(nuts_diag_lineno - 1)]
  } else if (length(nuts_nondiag_lineno) > 0) {  
    adaptation_info <- paste(comments[nuts_nondiag_lineno:len], collapse = '\n')
    comments <- comments[1:(nuts_nondiag_lineno - 1)]
    nondiag_mass <- 1
  }

  has_eq <- sapply(comments, function(i) grepl('=', i))
  comments <- comments[has_eq] 
  comments <- gsub('^#+\\s*|\\s*$', '', comments)
  eq_pos <- regexpr("=", comments, fixed = TRUE)
  names <- substr(comments, 0, eq_pos - 1)
  values <- as.list(substring(comments, eq_pos + 1))
  names(values) <- names
  values[['adaptation_info']] <- adaptation_info 
  names1 <- intersect(c("thin", "iter", "warmup", "equal_step_sizes", "chain_id",
                        "leapfrog_steps", "nondiag_mass",
                        "max_treedepth", "save_warmup"), names)
  names2 <- intersect(c("epsilon", "epsilon_pm", "gamma", "delta"), names) 
  for (z in names1) values[[z]] <- as.integer(values[[z]])
  for (z in names2) values[[z]] <- as.numeric(values[[z]])
  if (!"nondiag_mass" %in% names) values[["nondiag_mass"]] <- nondiag_mass
  values
}


read_stan_csv <- function(csvfiles) {
  # Read the csv files saved from Stan (or RStan) to a stanfit object
  # Args:
  #   csvfiles: csv files fitted for the same model; each file contains 
  #   the sample of one chain 
  # Assumptions:
  #   parameters in the CSV files are in order by row-major
  # 

  if (length(csvfiles) < 1) 
    stop("csvfiles does not contain any CSV file name")

  g_skip <- 10
  g_max_comm <- 50 # maximum number of lines of comments 
  ss_lst <- lapply(csvfiles, function(csv) read.csv(csv, header = TRUE, skip = 10, comment.char = '#'))
  cs_lst <- lapply(csvfiles, function(csv) read_comments(csv, n = g_max_comm))
  m_name <- sub("(_\\d+)*$", '', filename_rm_ext(csvfiles[1]))

  sdate <- do.call(max, lapply(csvfiles, function(csv) file.info(csv)$mtime))
  sdate <- format(sdate, "%a %b %d %X %Y") # same format as date() 

  chains <- length(ss_lst)
  fnames <- names(ss_lst[[1]])
  n_save <- nrow(ss_lst[[1]])
  paridx <- paridx_fun(fnames)
  lp__idx <- attr(paridx, 'meta')["lp__"]
  par_fnames <- c(fnames[paridx], "lp__")
  pars_oi <- unique_par(par_fnames)
  dims_oi <- lapply(pars_oi, 
                    function(i) {
                      pat <- paste('^', i, '(\\.\\d+)*$', sep = '')
                      i_fnames <- par_fnames[grepl(pat, par_fnames)]
                      get_dims_from_fnames(i_fnames, i) 
                    })
  names(dims_oi) <- pars_oi
  idx_2colm <- multi_idx_row2colm(dims_oi)
  if (chains > 1) {
    if (!all(sapply(ss_lst[-1], function(i) identical(names(i), fnames))))
      stop('the CSV files do not have same parameters')
    if (!all(sapply(ss_lst[-1], function(i) identical(length(i[[1]]), n_save)))) 
      stop('the number of iterations are not the same in all CSV files')
  } 

  cs_lst2 <- lapply(cs_lst, parse_stancsv_comments)

  samples <- lapply(ss_lst, 
                    function(df) {
                      ss <- df[c(paridx, lp__idx)[idx_2colm]]
                      attr(ss, "sampler_params") <- df[setdiff(attr(paridx, 'meta'), lp__idx)] 
                      ss
                    })
  par_fnames <- par_fnames[idx_2colm]
  for (i in seq_along(samples)) {
    attr(samples[[i]], "adaptation_info") <- cs_lst2[[i]]$adaptation_info 
    attr(samples[[i]], "args") <- 
      list(sampler = get_sampler_name(cs_lst2[[i]]$leapfrog_steps, cs_lst2[[i]]$equal_step_sizes, cs_lst2[[i]]$nondiag_mass),
           chain_id = cs_lst2[[i]]$chain_id)
  } 

  save_warmup <- sapply(cs_lst2, function(i) i$save_warmup)
  warmup <- sapply(cs_lst2, function(i) i$warmup)
  thin <- sapply(cs_lst2, function(i) i$thin)
  iter <- sapply(cs_lst2, function(i) i$iter)
  if (!all_int_eq(warmup) || !all_int_eq(thin) || !all_int_eq(iter)) 
    stop("not all iter/warmups/thin are the same in all CSV files")
  n_kept0 <- (iter[1] - 1) %/% thin[1] - (warmup[1] - 1) %/% thin[1]
  warmup2 <- 0
  if (max(save_warmup) == 0L) { # all equal to 0L
    n_kept <- n_save
  } else if (min(save_warmup) == 1L) { # all equals to 1L 
    warmup2 <- 1 + (warmup[1] - 1) %/% thin[1]
    n_kept <- n_save - warmup2 
  } 
  
  if (n_kept0 != n_kept) 
    stop("the number of iterations after warmup found (", n_kept, 
         ") does not match iter/warmup/thin from CSV comments (", n_kept0, ")")

  perm_lst <- lapply(1:chains, function(id) sample.int(n_kept))

  sim = list(samples = samples, 
             iter = iter[1], 
             thin = thin[1], 
             warmup = warmup[1], 
             chains = chains, 
             n_save = rep(n_save, chains),
             warmup2 = rep(warmup2, chains),
             permutation = perm_lst,
             pars_oi = pars_oi, 
             dims_oi = dims_oi,
             fnames_oi = dotfnames_to_sqrfnames(par_fnames), 
             n_flatnames = length(par_fnames))
  null_dso <- new("cxxdso", sig = list(character(0)), dso_saved = FALSE, dso_filename = character(0), 
                  modulename = character(0), system = R.version$system, cxxflags = character(0), 
                 .CXXDSOMISC = new.env())
  null_sm <- new("stanmodel", model_name = m_name, model_code = character(0), 
                 model_cpp = list(), dso = null_dso)

  nfit <- new("stanfit", 
              model_name = m_name,
              model_pars = pars_oi,
              par_dims = dims_oi, 
              mode = 0L,
              sim = sim,
              inits = list(), 
              stan_args = cs_lst2,
              stanmodel = null_sm,
              date = sdate, # not the time of sampling
              .MISC = new.env())
  invisible(nfit)
}
