filename_ext <- function(x) {
  # obtain the file extension 
  # copied from tools package 
  pos <- regexpr("\\.([[:alnum:]]+)$", x)
  ifelse(pos > -1L, substring(x, pos + 1L), "")
}

filename_rm_ext <- function(x) {
  # remove the filename's extension 
  sub("\\.[^.]*$", "", x)
} 

#   is_whole_number <- function(x) {
#     all.equal(x, round(x), check.attributes = FALSE) 
#   } 

#   as_integer_if_doable <- function(y) {
#     if (!is.numeric(y)) return(y) 
#     if (is.integer(y)) return(y) 
#     if (isTRUE(all.equal(y, round(y), check.attributes = FALSE))) 
#       storage.mode(y) <- "integer"  
#     return(y) 
#   } 
 
list_as_integer_if_doable <- function(x) {
  # change the storage mode from 'real' to 'integer' 
  # if applicable since by default R use real.
  #
  # Args:
  #  x: A list 
  # 
  # Note:
  # Ignore non-numeric vectors since we ignore
  # them in rlist_var_context 
  #
  lapply(x, 
         FUN = function(y) { 
           if (!is.numeric(y)) return(y) 
           if (is.integer(y)) return(y) 
           if (isTRUE(all.equal(y, round(y), check.attributes = FALSE))) 
             storage.mode(y) <- "integer"  
           return(y) 
         })
} 

mklist <- function(names, env = parent.frame()) { 
  # Make a list using names 
  # Args: 
  #   names: character strings of names of objects 
  #   env: the environment to look for objects with names
  # Note: we use inherits = TRUE when calling mget 
  #   and only mode of numeric and list are extracted (this
  #   is to avoid such as get a primitive function such as
  #   gamma.)
  d <- mget(names, env, ifnotfound = NA, inherits = TRUE, mode = 'numeric') 
  idx_is_na <- is.na(d)
  names_nf <- names[idx_is_na]
  if (length(names_nf) == 0) return(d)
  d2 <- mget(names_nf, env, ifnotfound = NA, inherits = TRUE, mode = 'list') 
  n <- which(is.na(d2))
  if (length(n) > 0) {
    stop(paste("objects ", paste("'", names_nf[n], "'", collapse = ', ', sep = ''), " not found", sep = ''))
  } 
  c(d[!idx_is_na],  d2)
} 

stan_kw1 <- c('for', 'in', 'while', 'repeat', 'until', 'if', 'then', 'else',
              'true', 'false') 
stan_kw2 <- c('int', 'real', 'vector', 'simplex', 'ordered', 'positive_ordered', 
              'row_vector', 'matrix', 'corr_matrix', 'cov_matrix', 'lower', 'upper') 
stan_kw3 <- c('model', 'data', 'parameters', 'quantities', 'transformed', 'generated') 

cpp_kw <- c("alignas", "alignof", "and", "and_eq", "asm", "auto", "bitand", "bitor", "bool", 
            "break", "case", "catch", "char", "char16_t", "char32_t", "class", "compl",
            "const", "constexpr", "const_cast", "continue", "decltype", "default", "delete",
            "do", "double", "dynamic_cast", "else", "enum", "explicit", "export", "extern",
            "false", "float", "for", "friend", "goto", "if", "inline", "int", "long", "mutable",
            "namespace", "new", "noexcept", "not", "not_eq", "nullptr", "operator", "or", "or_eq",
            "private", "protected", "public", "register", "reinterpret_cast", "return",
            "short", "signed", "sizeof", "static", "static_assert", "static_cast", "struct",
            "switch", "template", "this", "thread_local", "throw", "true", "try", "typedef",
            "typeid", "typename", "union", "unsigned", "using", "virtual", "void", "volatile",
            "wchar_t", "while", "xor", "xor_eq")


is_legal_stan_vname <- function(name) {
  # Return:
  #   FALSE: not a lega variable name in Stan 
  #   TRUE: maybe it is valid, but 100% sure 
  if (grepl('\\.',  name)) return(FALSE) 
  if (grepl('^\\d', name)) return(FALSE)
  if (grepl('__$',  name)) return(FALSE)
  if (name %in% stan_kw1) return(FALSE)
  if (name %in% stan_kw2) return(FALSE)
  if (name %in% stan_kw3) return(FALSE)
  !name %in% cpp_kw
} 

data_list2array <- function(x) {
  # Turn a list of array to an array whose first dimension is the list
  # and other dimensions being the dimensions of the array element.
  # So this would allow data in Stan coded as `vector[J] y[I]` 
  # to read data in form a list that has I elements of vector of length J, say 
  # 
  # # I <- 4; J <- 5
  # # y <- lapply(1:I, function(i) rnorm(J))
  # 
  # Args:
  #   x: A list of numeric array with the same dimensions 
  # Returns:
  #   An array with the first dimension indexes the list;
  #   other dimensions being the dimensions of the list element (an array)
  # 
  len <- length(x) 
  if (len == 0L)  return(NULL)

  dimx1 <- dim(x[[1]])

  if (any(sapply(x, function(xi) !is.numeric(xi))))  
    stop("all elements of the list should be numeric")
  if (is.null(dimx1)) dimx1 <- length(x[[1]]) 
  lendimx1 <- length(dimx1)

  if (len > 1) { 
    d <- sapply(x[-1], 
                function(xi) {
                  dimxi <- dim(xi)
                  if (is.null(dimxi)) dimxi <- length(xi)
                  identical(dimxi, dimx1) 
                })
    if (!all(d)) stop("the dimensions for all elements (array) of the list are not same")
  }

  # TODO(?): check if x is numeric or array. 
  x <- do.call(c, x)
  dim(x) <- c(dimx1, len)
  aperm(x, c(lendimx1 + 1L, seq_len(lendimx1)))
} 

data_preprocess <- function(data) { # , varnames) {
  # Preprocess the data (list or env) to list for stan
  # 
  # Args:
  #  data: A list, an environment, or a vector of character strings for names
  #  of objects 
  #   * stop if there is NA; no-name lists; duplicate names  
  #   * stop if the objects given name is not found  
  #   * remove NULL, non-numeric elements 
  #   * change to integers when applicable 

  # 
  # if (is.environment(data)) {
    
  #   data <- mget(varnames, envir = data, mode = "numeric", 
  #                ifnotfound = list(NULL))
  #   data <- data[!sapply(data, is.null)]
  # }
  if (is.environment(data)) {
    data <- as.list(data) 
  } else if (is.list(data)) {
    v <- names(data)
    if (is.null(v)) 
      stop("data must be a named list")
          
    ## Stan would report error if variable is not found 
    ## from the list
    # if (any(nchar(v) == 0))  
    #   stop("unnamed variables in data list")
    # 
 
    if (any(duplicated(v))) {
      stop("duplicated names in data list: ", 
           paste(v[duplicated(v)], collapse = " "))
    }
  } else {
    stop("data must be a list or an environment") 
  } 

  names <- names(data) 
  for (x in names) { 
    if (!is_legal_stan_vname(x))
    stop(paste('data with name ', x, " is not allowed in Stan", sep = ''))
  } 
 
  data <- lapply(names, 
                 FUN = function(name) {
                   x <- data[[name]]
                   if (is.data.frame(x)) { 
                     x <- data.matrix(x) # change data.frame to array 
                   } else if (is.list(x)) {
                     x <- data_list2array(x) # list to array
                   } 
 
                   ## Now we stop whenever we have NA in the data
                   ## since we do not know what variables are needed
                   ## at this point.
                   if (any(is.na(x))) {
                     stop("Stan does not support NA (in ", name, ") in data")
                   } 
 
                   # remove those not numeric data 
                   if (!is.numeric(x)) {
                     warning("data with name ", name, " is not numeric and not used")
                     return(NULL) 
                   }
 
                   if (is.integer(x)) return(x) 
         
                   # change those integers stored as reals to integers 
                   if (isTRUE(all.equal(x, round(x), check.attributes = FALSE))) 
                     storage.mode(x) <- "integer"  
                   return(x) 
                 })   
 
  names(data) <- names
  data[!sapply(data, is.null)] 
} 


read_model_from_con <- function(con) {
  lines <- readLines(con, n = -1L, warn = FALSE)
  paste(lines, collapse = '\n') 
} 

get_model_strcode <- function(file, model_code = '') {
  # return the model code as a character string 
  # Args:
  #   file: a file or connection 
  #   model_code: character string for one of the following
  #     * the name of an object of character string 
  #     * the model code itself 
  # 
  # Returns: 
  #   the model code with attribute model_name2,
  #   a name implied from file or object name,
  #   which can be used later when model_name is not 
  #   specified for say for function stan. 

  if (!missing(file)) {
    if (is.character(file)) {
      fname <- file
      model_name2 <- sub("\\.[^.]*$", "", filename_rm_ext(basename(fname))) 
      file <- try(file(fname, "rt"))
      if (inherits(file, "try-error")) {
        stop(paste("cannot open model file \"", fname, "\"", sep = ""))
      }
      on.exit(close(file))
    } else if (!inherits(file, "connection")) {
      stop("file must be a character string or connection")
    }
    model_code <- paste(readLines(file, warn = FALSE), collapse = '\n') 
    # the model name implied from file name, which
    # will be used if model_name is not specified later
    attr(model_code, "model_name2") <- model_name2 
    return(model_code) 
  }

  model_name2 <- attr(model_code, "model_name2") 
  if (is.null(model_name2)) 
    model_name2 <- deparse(substitute(model_code))
  if (model_code != '' && is.character(model_code)) {  
    if (!grepl("\\{", model_code)) {
      # model_code points an object that includes the model 
      model_name2 <- model_code
      if (!exists(model_code, mode = 'character', envir = parent.frame())) 
        stop(paste("cannot find ", model_code, sep = '')) 
      model_code <- get(model_code, mode = 'character', envir = parent.frame()) 
    } else {
      # model_code includes the code itself, two cases of passing:
      #  1. using another object such as stan(mode_code = scode)`
      #  2. providing the string directly such stan(model_code = "")
      if (grepl("\\{", model_name2)) 
        model_name2 <- 'anon_model' 
    } 
    attr(model_code, "model_name2") <- model_name2 
    return(model_code) 
  } 

  stop("model file missing and empty model_code")
} 

# FIXEME: implement more check on the arguments 
check_args <- function(argss) {
  if (FALSE) stop() 
} 

#
# model_code <- read_model_from_con('http://stan.googlecode.com/git/src/models/bugs_examples/vol1/dyes/dyes.stan')
# cat(model_code)


append_id <- function(file, id, suffix = '.csv') {
  fname <- basename(file)
  fpath <- dirname(file)
  fname2 <- gsub("\\.csv[[:space:]]*$", 
                 paste("_", id, ".csv", sep = ''), 
                 fname)
  if (fname2 == fname) 
    fname2 <- paste(fname, "_", id, ".csv", sep = '')
  file.path(fpath, fname2)
}

check_seed <- function(seed, warn = 0) {
  if (is.character(seed) && grepl("[^0-9]", seed)) {
    if (warn == 0) stop("seed needs to be string of digits")
    else message("seed needs to be string of digits")
    return(NULL)
  } 
  if (is.numeric(seed)) seed <- as.integer(seed)
  if (is.na(seed)) seed <- sample.int(.Machine$integer.max, 1)
  seed 
} 

config_argss <- function(chains, iter, warmup, thin, 
                         init, seed, sample_file, ...) {

  iter <- as.integer(iter) 
  if (iter < 1) 
    stop("parameter 'iter' should be a positive integer")
  thin <- as.integer(thin) 
  if (thin < 1 || thin > iter) 
    stop("parameter 'thin' should be a positive integer less than 'iter'")
  warmup <- max(0, as.integer(warmup)) 
  if (warmup > iter) 
    stop("parameter 'warmup' should be an integer less than 'iter'")
  chains <- as.integer(chains) 
  if (chains < 1) 
    stop("parameter 'chains' should be a positive integer")

  iters <- rep(iter, chains)   
  thins <- rep(thin, chains)  
  warmups <- rep(warmup, chains) 

  inits_specified <- FALSE
  if (is.numeric(init)) init <- as.character(init) 
  if (is.character(init)) {
    if (init[1] %in% c("0", "random")) inits <- rep(init[1], chains) 
    else inits <- rep("random", chains) 
    inits_specified <- TRUE
  } 
  if (!inits_specified && is.function(init)) {
    ## the function can take an argument named by chain_id 
    if (any(names(formals(init)) == "chain_id")) {
      inits <- lapply(1:chains, function(id) init(chain_id = id))
    } else {
      inits <- lapply(1:chains, function(id) init())
    } 
    inits_specified <- TRUE
  } 
  if (!inits_specified && is.list(init)) {
    if (length(init) != chains) 
      stop("initial value list mismatchs number of chains") 
    if (!any(sapply(init, is.list))) {
      stop("initial value list is not a list of lists") 
    }
    inits <- init; 
    inits_specified <- TRUE
  }
  if (!inits_specified) stop("wrong specification of initial values")

  ## only one seed is needed by virtue of the RNG 
  seed <- if (missing(seed)) sample.int(.Machine$integer.max, 1) else check_seed(seed)

  dotlist <- list(...)
  dotlist$point_estimate <- FALSE # not to do point estimation

  # use chain_id argument if specified
  if (is.null(dotlist$chain_id)) { 
    chain_ids <- seq_len(chains)
  } else {
    chain_id <- as.integer(dotlist$chain_id)
    if (any(duplicated(chain_id))) stop("chain_id has duplicated elements")
    chain_id_len <- length(chain_id)
    chain_ids <- if (chain_id_len >= chains) chain_id else {
                   c(chain_id, max(chain_id) + seq_len(chains - chain_id_len))
                 }
    dotlist$chain_id <- NULL
  }

  argss <- vector("list", chains)  
  ## the name of arguments in the list need to 
  ## match those in include/rstan/stan_args.hpp 
  for (i in 1:chains)  
    argss[[i]] <- list(chain_id = chain_ids[i],
                       iter = iters[i], thin = thins[i], seed = seed, 
                       warmup = warmups[i], init = inits[[i]]) 
    
  if (!missing(sample_file) && !is.na(sample_file)) {
    sample_file <- writable_sample_file(sample_file) 
    if (chains == 1) 
        argss[[1]]$sample_file <- sample_file
    if (chains > 1) {
      for (i in 1:chains) 
        argss[[i]]$sample_file <- append_id(sample_file, i) 
    }
  }
  
  for (i in 1:chains)
    argss[[i]] <- c(argss[[i]], dotlist)
  check_args(argss)
  argss 
} 

is_dir_writable <- function(path) {
  (file.access(path, mode = 2) == 0) && (file.access(path, mode = 1) == 0)  
} 

writable_sample_file <- 
function(file, warn = TRUE, 
         wfun = function(x, x2) {
           paste('"', x, '" is not writable; use "', x2, '" instead', sep = '')
         }) { 
  # Check if the path for file is writable, if not using tempdir() 
  # 
  # Args:
  #  file: The file interested. 
  #  warning: TRUE give a warning. 
  #  warningfun: A function that take two dirs for creating 
  #    the warning message. 
  # 
  # Returns:
  #  If the specified file is writable, return itself. 
  #  Otherwise, change the path to tempdir(). 
  
  dir <- dirname(file) 
  if (is_dir_writable(dir)) return(file)  

  dir2 <- tempdir()
  if (warn) warning(wfun(dir, dir2))
  file.path(dir2, basename(file))
} 


#   probs2str <- function(probs, digits = 1) {
#     paste(formatC(probs * 100,  
#                   digits = digits, 
#                   format = 'f', 
#                   drop0trailing = TRUE), 
#           "%", sep = '')
#   } 

stan_rdump <- function(list, file = "", append = FALSE, 
                       envir = parent.frame(),
                       width = options("width")$width, quiet = FALSE) {
  # Dump an R list or environment for a model data 
  # to the R dump file that Stan supports.
  #
  # Args:
  #   list: a vector of character for all variables interested 
  #         (the same as in R's dump function) 
  #   file: the output file for dumping the variables. 
  #   append: then TRUE, the file is opened with 
  #           mode of appending; otherwise, a new file
  #           is created.  
  #   quiet: no warning if TRUE
  # 
  # Return:
 
  if (is.character(file)) {
    ex <- sapply(list, exists, envir = envir)
    if (!all(ex)) {
      notfound_list <- list[!ex] 
      if (!quiet) 
        warning(paste("objects not found: ", paste(notfound_list, collapse = ', '), sep = '')) 
    } 
    list <- list[ex] 
    if (!any(ex)) 
      return(invisible(character()))

    if (nzchar(file)) {
      file <- file(file, ifelse(append, "a", "w"))
      on.exit(close(file), add = TRUE)
    } else {
      file <- stdout()
    }
  }

  for (x in list) { 
    if (!is_legal_stan_vname(x) & !quiet)
      warning(paste("variable name ", x, " is not allowed in Stan", sep = ''))
  } 

  l2 <- NULL
  addnlpat <- paste0("(.{1,", width, "})(\\s|$)")
  for (v in list) {
    vv <- get(v, envir) 

    if (is.data.frame(vv)) {
      vv <- data.matrix(vv) 
    } else if (is.list(vv)) {
      vv <- data_list2array(vv)
    } else if (is.logical(vv)) {
      mode(vv) <- "integer"
    } else if (is.factor(vv)) {
      vv <- as.integer(vv)
    } 
    
    if (!is.numeric(vv))  {
      if (!quiet) 
        warning(paste0("variable ", v, " is not supported for dumping."))
      next
    } 

    if (is.vector(vv)) {
      if (length(vv) == 1) {
        cat(v, " <- ", vv, "\n", file = file, sep = '')
        next
      }
      str <- paste0(v, " <- \nc(", paste(vv, collapse = ', '), ")") 
      str <-  gsub(addnlpat, '\\1\n', str)
      cat(str, file = file) 
      l2 <- c(l2, v) 
      next
    }    

    if (is.matrix(vv) || is.array(vv)) { 
      l2 <- c(l2, v) 
      vvdim <- dim(vv)
      cat(v, " <- \n", file = file, sep = '')
      str <- paste0("structure(c(", paste(as.vector(vv), collapse = ', '), "),") 
      str <- gsub(addnlpat, '\\1\n', str)
      cat(str, 
          ".Dim = c(", paste(vvdim, collapse = ', '), "))\n", file = file, sep = '')
      next
    }
  }
  invisible(l2) 
} 

## test stan_rdump simply
# a <- 1:3
# b <- 3
# c <- matrix(1:9, ncol = 3)
# d <- array(1:90, dim = c(9, 2, 5))
# stan_rdump(c('a', 'b', 'c', 'd'), file = 'a.txt')

get_rhat_cols <- function(rhats) {
  # 
  # Args:
  #   rhats: a vector of rhats
  #
  rhat_nan_col <- rstan_options("plot_rhat_nan_col")
  rhat_large_col <- rstan_options("plot_rhat_large_col")
  rhat_breaks <- rstan_options("plot_rhat_breaks")
  # print(rhat_breaks)
  rhat_colors <- rstan_options("plot_rhat_cols")

  sapply(rhats, 
         FUN = function(x) {
           if (is.na(x) || is.nan(x) || is.infinite(x))
             return(rhat_nan_col)           
           for (i in 1:length(rhat_breaks)) {
             # cat("i=", i, "\n")
             if (x >= rhat_breaks[i]) next
             return(rhat_colors[i])
           }  
           rhat_large_col
         })  
}

plot_rhat_legend <- function(x, y, cex = 1) { 
  # Args
  #   x, y: left, bottom corner coordinates 
  #   cex: cex for the labels 
  rhat_breaks <- rstan_options("plot_rhat_breaks")
  n_breaks <- length(rhat_breaks) 
  rhat_colors <- rstan_options("plot_rhat_cols")[1:n_breaks] 
  rhat_legend_labels <- c(paste("< ", rhat_breaks, "  ", sep = ''), 
                        paste(">= ", max(rhat_breaks), "  ", sep = ''),
                        "NaN/Inf")
  rhat_legend_cols <- c(rhat_colors, rstan_options('plot_rhat_large_col'),
                        rstan_options("plot_rhat_nan_col"))
  rhat_legend_width <- strwidth(rhat_legend_labels, cex = cex) 
  rhat_rect_width <- strwidth("r-hat ", cex = cex) 
  text(x, y, label = 'Rhat:  ', adj = c(0, 0), cex = cex)  
  s1 <- strwidth('Rhat:  ', cex = cex) 
  starts <- x + c(s1, s1 + cumsum(rhat_rect_width + rhat_legend_width)) 

  height <- strheight("0123456789<>=", cex = cex)

  for (i in 1:length(rhat_legend_cols)) {
    rect(starts[i], y, starts[i] + rhat_rect_width, y + height, col = rhat_legend_cols[i], border = NA) 
    text(starts[i] + rhat_rect_width, y, adj = c(0, 0), label = rhat_legend_labels[i], cex = cex) 
  } 
} 
  

read_rdump <- function(f) {
  # Read data defined in an R dump file to an R list
  # 
  # Args:
  #   f: the file to be sourced
  # 
  # Returns:
  #   A list

  if (missing(f)) 
    stop("no file specified.")
  e <- new.env() 
  source(file = f, local = e)
  as.list(e)
} 


idx_col2rowm <- function(d) {
  # Suppose an iteration of samples for an array parameter is ordered by
  # col-major. This function generates the indexes that can be used to change
  # the sequences to row-major. 
  # Args:
  #   d: the dimension of the parameter 
  len <- length(d) 
  if (0 == len) return(1)  
  if (1 == len) return(1:d)  
  idx <- aperm(array(1:prod(d), dim = d)) 
  return(as.vector(idx)) 
} 


idx_row2colm <- function(d) {
  # What if it is row-major and we want col_major? 
  len <- length(d) 
  if (0 == len) return(1)  
  if (1 == len) return(1:d)  
  idx <- aperm(array(1:prod(d), dim = rev(d))) 
  return(as.vector(idx)) 
} 

multi_idx_row2colm <- function(dims) {
  # Suppose we want to change a vector of parameter names (each of which is in
  # row major) to col major.  This function serves to get the indexes.    
  # Args:
  #   dims: a list of dimensions for all the parameters
  #  
  ## print(dims)
  shifts <- calc_starts(dims) - 1
  idx <- lapply(seq_along(shifts), function(i) shifts[i] + idx_row2colm(dims[[i]]))
  do.call(c, idx)
} 


seq_array_ind <- function(d, col_major = FALSE) {
  #
  # Generate an array of indexes for an array parameter 
  # in order of major or column. 
  #
  # Args:
  #   d: the dimensions of an array parameter, for example, 
  #     c(2, 3). 
  # 
  #   col_major: Determine what is the order of indexes. 
  #   If col_major = TRUE, for d = c(2, 3), return 
  #   [1, 1] 
  #   [2, 1] 
  #   [1, 2] 
  #   [2, 2] 
  #   [1, 3] 
  #   [2, 3] 
  #   If col_major = FALSE, for d = c(2, 3), return 
  #   [1, 1] 
  #   [1, 2] 
  #   [1, 3] 
  #   [2, 1] 
  #   [2, 2] 
  #   [2, 3] 
  # 
  # Returns: 
  #   If length of d is 0, return empty vector. 
  #   Otherwise, return an array of indexes, each
  #   row of which is an index. 
  # 
  # Note:
  #   R function arrayInd might be helpful sometimes. 
  # 
  if (length(d) == 0L)
    return(numeric(0L)) 
  total <- prod(d) 
  len <- length(d) 
  res <- array(1L, dim = c(total, len)) 
  jidx <- if (col_major) 1L:len else len:1L
  for (i in 2L:total) {
    res[i, ] <- res[i - 1, ]
    for (j in jidx) { 
      if (res[i - 1, j] < d[j]) {
        res[i, j] <- res[i - 1, j] + 1
        break
      } 
      res[i, j] <- 1
    } 
  } 
  res 
} 

flat_one_par <- function(n, d, col_major = FALSE) {
  # Return all the elemenetwise parameters for a vector/array
  # parameter. 
  # 
  # Args:
  #  n: Name of the parameter. For example, n = "alpha" 
  #  d: A vector indicates the dimensions of parameter n. 
  #     For example, d = c(2, 3).  d could be empty 
  #     as well when n is a scalar. 
  # 
  if (0 == length(d)) return(n)  
  nameidx <- seq_array_ind(d, col_major) 
  names <- apply(nameidx, 1, function(x) paste(n, "[", paste(x, collapse = ','), "]", sep = '')) 
  as.vector(names) 
} 


flatnames <- function(names, dims, col_major = FALSE) {
  if (length(names) == 1) 
    return(flat_one_par(names, dims[[1]], col_major = col_major))  
  nameslst <- mapply(flat_one_par, names, dims, 
                     MoreArgs = list(col_major = col_major), 
                     SIMPLIFY = FALSE,
                     USE.NAMES = FALSE) 
  if (is.vector(nameslst, "character")) 
    return(nameslst) 
  do.call(c, nameslst) 
} 

num_pars <- function(d) prod(d) 

calc_starts <- function(dims) {
  len <- length(dims) 
  s <- sapply(dims, function(d)  num_pars(d), USE.NAMES = FALSE) 
  cumsum(c(1, s))[1:len] 
} 

check_pars <- function(allpars, pars) {
  pars_wo_ws <- gsub('\\s+', '', pars) 
  m <- which(match(pars_wo_ws, allpars, nomatch = 0) == 0)
  if (length(m) > 0) 
    stop("no parameter ", paste(pars[m], collapse = ', ')) 
  if (length(pars_wo_ws) == 0) 
    stop("no parameter specified (pars is empty)")
  unique(pars_wo_ws) 
} 

check_pars_first <- function(object, pars) {
  # Check if all parameters in pars are valid parameters of the model 
  # Args:
  #   object: a stanfit object 
  #   pars: a character vector of parameter names
  # Returns:
  #   pars without white spaces, if any, if all are valid
  #   otherwise stop reporting error
  allpars <- cbind(object@model_pars, flatnames(object@model_pars))
  check_pars(allpars, pars) 
} 

check_pars_second <- function(sim, pars) {
  #
  # Check if all parameters in pars are parameters for which we saved
  # their samples 
  # 
  # Args:
  #   sim: The sim slot of class stanfit 
  #   pars: a character vector of parameter names
  # 
  # Returns:
  #   pars without white spaces, if any, if all are valid
  #   otherwise stop reporting error
  if (missing(pars)) return(sim$pars_oi) 
  allpars <- c(sim$pars_oi, sim$fnames_oi) 
  check_pars(allpars, pars)
} 

pars_total_indexes <- function(names, dims, fnames, pars) {
  # Obtain the total indexes for parameters (pars) in the 
  # whole sequences of names that is order by 'column major.' 
  # Args: 
  #   names: all the parameters names specifying the sequence of parameters
  #   dims:  the dimensions for all parameters, the order for all parameters 
  #          should be the same with that in 'names'
  #   fnames: all the parameter names specified by names and dims 
  #   pars:  the parameters of interest. This function assumes that
  #     pars are in names.   
  # Note: inside each parameter (vector or array), the sequence is in terms of
  #   col-major. That means if we have parameter alpha and beta, the dims
  #   of which are [2,2] and [2,3] respectively.  The whole parameter sequence
  #   are alpha[1,1], alpha[2,1], alpha[1,2], alpha[2,2], beta[1,1], beta[2,1],
  #   beta[1,2], beta[2,2], beta[1,3], beta[2,3]. In addition, for the col-majored
  #   sequence, an attribute named 'row_major_idx' is attached, which could
  #   be used when row major index is favored.

  starts <- calc_starts(dims) 
  par_total_indexes <- function(par) {
    # for just one parameter
    #
    p <- match(par, fnames)
    # note that here when `par' is a scalar, it would
    # match to one of `fnames'
    if (!is.na(p)) {
      names(p) <- par 
      attr(p, "row_major_idx") <- p 
      return(p) 
    } 
    p <- match(par, names) 
    idx <- starts[p] + seq(0, by = 1, length.out = num_pars(dims[[p]])) 
    names(idx) <- fnames[idx] 
    attr(idx, "row_major_idx") <- starts[p] + idx_col2rowm(dims[[p]]) - 1 
    idx
  } 
  idx <- lapply(pars, FUN = par_total_indexes) 
  names(idx) <- pars 
  idx 
} 

## simple test for pars_total_indexes 
#  names <- c('alpha', 'beta', 'gamma')
#  dims <- list(c(2,3), c(3,4,5), c(5))
#  fnames <- flatnames(names, dims, col_major = TRUE) 
#  pars_total_indexes(names, dims, fnames, c('gamma', 'alpha', 'beta')) 

#### temporary test code 
#  a <- config_argss(3, c(100, 200), 10, 1, "user", NULL, seed = 3) 
#  print(a) 
#  
#  fun1 <- function(chain_id) {
#    cat("chain_id=", chain_id)
#    return(list(mu = chain_id))
#  } 
#  b <- config_argss(3, c(100, 200), 10, 1, c("user", 1), fun1, seed = 3) 
#  print(b)
#  
#  
rstancolgrey <- rgb(matrix(c(247, 247, 247, 204, 204, 204, 150, 150, 150, 82, 82, 82),  
                           byrow = TRUE, ncol = 3), 
                    alpha = 100, 
                    names = paste(1:4), maxColorValue = 255)

# from http://colorbrewer2.org/, colorblind safe, 
# 6 different colors, diverging 
rstancolc <- rgb(matrix(c(230, 97, 1, 
                          153, 142, 195, 
                          84, 39, 136, 
                          241, 163, 64, 
                          216, 218, 235, 
                          254, 224, 182), 
                        byrow = TRUE, ncol = 3),
                 names = paste(1:6), maxColorValue = 255) 

default_summary_probs <- function() c(0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975)

## summarize the chains merged and individually 
get_par_summary <- function(sim, n, probs = default_summary_probs()) {
  ss <- lapply(1:sim$chains, function(i) sim$samples[[i]][[n]][-(1:sim$warmup2[i])]) 
  msdfun <- function(chain) c(mean(chain), sd(chain))
  qfun <- function(chain) quantile(chain, probs = probs)
  c_msd <- unlist(lapply(ss, msdfun), use.names = FALSE) 
  c_quan <- unlist(lapply(ss, qfun), use.names = FALSE) 
  ass <- do.call(c, ss) 
  msd <- msdfun(ass) 
  quan <- qfun(ass) 
  list(msd = msdfun(ass), quan = qfun(ass), c_msd = c_msd, c_quan = c_quan) 
} 

# mean and sd 
get_par_summary_msd <- function(sim, n) { 
  ss <- lapply(1:sim$chains, function(i) sim$samples[[i]][[n]][-(1:sim$warmup2[i])]) 
  sumfun <- function(chain) c(mean(chain), sd(chain)) 
  cs <- lapply(ss, sumfun)
  as <- sumfun(do.call(c, ss)) 
  list(msd = as, c_msd = unlist(cs, use.names = FALSE)) 
} 

# quantiles 
get_par_summary_quantile <- function(sim, n, probs = default_summary_probs()) {
  ss <- lapply(1:sim$chains, function(i) sim$samples[[i]][[n]][-(1:sim$warmup2[i])]) 
  sumfun <- function(chain) quantile(chain, probs = probs)
  cs <- lapply(ss, sumfun)
  as <- sumfun(do.call(c, ss)) 
  list(quan = as, c_quan = unlist(cs, use.names = FALSE)) 
} 

combine_msd_quan <- function(msd, quan) {
  # Combine msd and quantiles for chain's summary 
  # Args:
  #   msd: the array for mean and sd with dim num.par * 2 * chains 
  #   cquan: the array for quantiles with dim num.par * n.quan * chains 
  dim1 <- dim(msd) 
  dim2 <- dim(quan) 
  if (any(dim1[c(1, 3)] != dim2[c(1, 3)])) 
    stop("numers of parameter/chains differ in msd and quan") 
  chains <- dim1[3] 
  n_par <- dim1[1] 
  n_stat <- dim1[2] + dim2[2] 
  par_names <- dimnames(msd)[[1]] 
  stat_names <- c(dimnames(msd)[[2]], dimnames(quan)[[2]]) 
  chain_id_names <- dimnames(msd)[[3]]
  fun <- function(i) {
    # This is a bit ugly; one reason is that we need to 
    # deal with the case that dim1[1] = 1, in which 
    # a1 is a vector. 
    a1 <- msd[, , i] 
    a2 <- quan[, , i] 
    dim(a1) <- dim1[1:2] 
    dim(a2) <- dim2[1:2] 
    cbind(a1, a2)
  } 
  ll <- lapply(1:chains, fun) 
  twodnames <- dimnames(ll[[1]]) 
  msdquan <- array(unlist(ll), dim = c(n_par, n_stat, chains)) 
  dimnames(msdquan) <- list(parameter = par_names, stats = stat_names, 
                            chains = chain_id_names) 
  msdquan 
} 


summary_sim <- function(sim, pars, probs = default_summary_probs()) {
  # cat("summary_sim is called.\n")
  probs_len <- length(probs) 
  pars <- if (missing(pars)) sim$pars_oi else check_pars_second(sim, pars) 
  tidx <- pars_total_indexes(sim$pars_oi, sim$dims_oi, sim$fnames_oi, pars) 
  tidx_rowm <- lapply(tidx, function(x) attr(x, "row_major_idx"))
  tidx <- unlist(tidx, use.names = FALSE)
  tidx_len <- length(tidx) 
  tidx_rowm <- unlist(tidx_rowm, use.names = FALSE)
  lmsdq <- lapply(tidx, function(n) get_par_summary(sim, n, probs)) 
  msd <- do.call(rbind, lapply(lmsdq, function(x) x$msd)) 
  quan <- do.call(rbind, lapply(lmsdq, function(x) x$quan)) 
  probs_str <- colnames(quan)
  dim(msd) <- c(tidx_len, 2) 
  dim(quan) <- c(tidx_len, probs_len) 
  rownames(msd) <- sim$fnames_oi[tidx] 
  rownames(quan) <- sim$fnames_oi[tidx] 
  colnames(msd) <- c("mean", "sd") 
  colnames(quan) <- probs_str 

  c_msd <- do.call(rbind, lapply(lmsdq, function(x) x$c_msd)) 
  c_quan <- do.call(rbind, lapply(lmsdq, function(x) x$c_quan)) 
  dim(c_msd) <- c(tidx_len, 2, sim$chains) 
  dim(c_quan) <- c(tidx_len, probs_len, sim$chains) 

  sim_attr_args <- attr(sim, "args")
  cids <- if (is.null(sim_attr_args)) {
            cids <- 1:sim$chains 
          } else {
            sapply(attr(sim, "args"), function(x) x$chain_id)
          }

  dimnames(c_msd) <- list(parameters = sim$fnames_oi[tidx], 
                          stats = c("mean", "sd"), 
                          chains = paste0("chain:", cids)) 
  dimnames(c_quan) <- list(parameters = sim$fnames_oi[tidx], 
                           stats = probs_str, 
                           chains = paste0("chains:", cids))

  ess <-  array(sapply(tidx, function(n) rstan_ess(sim, n)), dim = c(tidx_len, 1)) 
  rhat <- array(sapply(tidx, function(n) rstan_splitrhat(sim, n)), dim = c(tidx_len, 1)) 

  ss <- list(msd = msd, sem = msd[, 2] / sqrt(ess), 
             c_msd = c_msd, quan = quan, c_quan = c_quan, 
             ess = ess, rhat = rhat) 
  attr(ss, "row_major_idx") <- tidx_rowm 
  attr(ss, "col_major_idx") <- tidx
  ss
}  

summary_sim_quan <- function(sim, pars, probs = default_summary_probs()) {
  # cat("summary_sim is called.\n")
  probs_len <- length(probs) 
  pars <- if (missing(pars)) sim$pars_oi else check_pars_second(sim, pars) 
  tidx <- pars_total_indexes(sim$pars_oi, sim$dims_oi, sim$fnames_oi, pars) 
  tidx_rowm <- lapply(tidx, function(x) attr(x, "row_major_idx"))
  tidx <- unlist(tidx, use.names = FALSE)
  tidx_len <- length(tidx) 
  tidx_rowm <- unlist(tidx_rowm, use.names = FALSE)
  lquan <- lapply(tidx, function(n) get_par_summary_quantile(sim, n, probs)) 
  quan <- do.call(rbind, lapply(lquan, function(x) x$quan)) 
  probs_str <- colnames(quan) 
  dim(quan) <- c(tidx_len, probs_len) 
  rownames(quan) <- sim$fnames_oi[tidx] 
  colnames(quan) <- probs_str 

  sim_attr_args <- attr(sim, "args")
  cids <- if (is.null(sim_attr_args)) {
            cids <- 1:sim$chains 
          } else {
            sapply(attr(sim, "args"), function(x) x$chain_id)
          }

  c_quan <- do.call(rbind, lapply(lquan, function(x) x$c_quan)) 
  dim(c_quan) <- c(tidx_len, probs_len, sim$chains) 
  dimnames(c_quan) <- list(parameters = sim$fnames_oi[tidx], 
                           stats = probs_str, 
                           chains = paste0("chains:", cids))

  ss <- list(quan = quan, c_quan = c_quan)
  attr(ss, "row_major_idx") <- tidx_rowm 
  attr(ss, "col_major_idx") <- tidx
  ss
}  

summary_sim_ess <- function(sim, pars) {
  pars <- if (missing(pars)) sim$pars_oi else check_pars_second(sim, pars) 
  tidx <- pars_total_indexes(sim$pars_oi, sim$dims_oi, sim$fnames_oi, pars) 
  tidx_rowm <- lapply(tidx, function(x) attr(x, "row_major_idx"))
  tidx <- unlist(tidx, use.names = FALSE)
  tidx_rowm <- unlist(tidx_rowm, use.names = FALSE)
  ess <- sapply(tidx, function(n) rstan_ess(sim, n)) 
  names(ess) <- sim$fnames_oi[tidx]
  attr(ess, "row_major_idx") <- tidx_rowm
  attr(ess, "col_major_idx") <- tidx
  ess
} 

summary_sim_rhat <- function(sim, pars) {
  pars <- if (missing(pars)) sim$pars_oi else check_pars_second(sim, pars) 
  tidx <- pars_total_indexes(sim$pars_oi, sim$dims_oi, sim$fnames_oi, pars) 
  tidx_rowm <- lapply(tidx, function(x) attr(x, "row_major_idx"))
  tidx <- unlist(tidx, use.names = FALSE)
  tidx_rowm <- unlist(tidx_rowm, use.names = FALSE)
  rhat <- sapply(tidx, function(n) rstan_splitrhat(sim, n)) 
  names(rhat) <- sim$fnames_oi[tidx]
  attr(rhat, "row_major_idx") <- tidx_rowm
  attr(rhat, "col_major_idx") <- tidx
  rhat 
} 


par_vector2list <- function(v, pars, dims, starts = calc_starts(dims)) {
  # Turn a vector of sample (typically an iteration)
  # into a list according to the dims for parameters
  # Args:
  #   v: the vector of sample 
  #   pars: a character vector for parameter names 
  #   dims: a list of integer vector for parameter dimensions
  lst <- lapply(seq_along(pars), 
                function(i) { 
                  len <- num_pars(dims[[i]]) 
                  y <- v[starts[i] + (1:len) - 1] 
                  if (length(dims[[i]]) > 0) dim(y) <- dims[[i]] 
                  return(y) 
                })
  names(lst) <- pars 
  lst 
}

organize_inits <- function(inits, pars, dims) {
  # obtain a list of inital values for each chain in sim
  # Args: 
  #   inits: a list of vectors, each vector is the 
  #     inits for a chain 
  #   pars: vector of character for the names 
  #   dims: a list of lists with equal length of `pars` 

  # remove element 'lp__' in the names 
  idx_of_lp <- which(pars == "lp__")
  if (idx_of_lp > 0) {
    pars <- pars[-idx_of_lp] 
    dims <- dims[-idx_of_lp] 
  }
  starts <- calc_starts(dims) 
  tmpfun <- function(x) par_vector2list(x, pars, dims, starts) 
  lapply(inits, tmpfun) 
} 

# ported from bugs.plot.inferences in R2WinBUGS  
# 
stan_plot_inferences <- function(sim, summary, pars, model_info, display_parallel = FALSE, ...) {
  # 
  # Args:
  #   sim: the sim list in stanfit object
  #   pars: parameters of interest
  #   model_info: names list with elements model_name and model_date 
  #   display_parallel

  alert_col <- rstan_options("rstan_alert_col")
  chain_cols <- rstan_options("rstan_chain_cols")
  chain_cols.len <- length(chain_cols) 

  # FIXME: the following if - else for all platforms 
  if (exists('windows'))  dev.fun <- windows 
  if (exists('X11'))  dev.fun <- X11 
  opt.dev <- options("device") 
  if (.Device %in% c("windows", "X11cairo")  ||
      (.Device=="null device" && identical(opt.dev, dev.fun))) {
    cex.points <- .7
    min.width <- .02
  } else {
    cex.points <- .3
    min.width <- .01
  }

  cex_names <- .7
  cex.axis <- .6 
  cex_tiny <- .4 
  # the standard number of parameters in an array parameters. 
  # we have this so that even the # of parameters are less than
  # 30, we still have equal space between parameters for 
  # the whole plot. 
  standard_width <- rstan_options('plot_standard_npar') 
  max_width <- rstan_options('plot_max_npar') 

  pars <- if (missing(pars)) sim$pars_oi else check_pars_second(sim, pars) 
  n_pars <- length(pars) 
  chains <- sim$chains
 
  tidx <- pars_total_indexes(sim$pars_oi, sim$dims_oi, sim$fnames_oi, pars) 

  ## if in Splus, suppress printing of warnings during the plotting.
  ## otherwise a warning is generated 
  if (!is.R()) {
    warn.settings <- options("warn")[[1]]
    options (warn = -1)
  }
  height <- .6
  # mar: c(bottom, left, top, right)
  par.old <- par(no.readonly = TRUE)
  on.exit(par(par.old)) 
  par(mar = c(1, 0, 1, 0))

  plot(c(0, 1), c(-n_pars - .5, -.4), 
       ann = FALSE, bty = "n", xaxt = "n", yaxt = "n", type = "n")
  if (!is.R())
    options(warn = warn.settings)

  # plot the model general information 
  header <- paste("Stan model '", model_info$model_name, "' (", chains, 
                  " chains: iter=", sim$iter, "; warmup=", sim$warmup, 
                  "; thin=", sim$thin, ") fitted at ",
                  model_info$model_date, sep = '') 
  # side: (1=bottom, 2=left, 3=top, 4=right)
  mtext(header, side = 3, outer = TRUE, line = -1, cex = .7)

  W <- max(strwidth(pars, cex = cex_names))
  # the max width of the variable names 

  # cex_names is defined at the beginning of this fun
  B <- (1 - W) / 3.8
  A <- 1 - 3.5 * B
  title <- if (display_parallel) "80% interval for each chain" else  "medians and 80% intervals"
  text(A, -.4, title, adj = 0, cex = cex_names)
  num_height <- strheight (1:9, cex = cex_tiny) * 1.2

  truncated <- FALSE 
  for (k in 1:n_pars) { 
    text (0, -k, pars[k], adj = 0, cex = cex_names)

    k_dim <- sim$dims_oi[[pars[k]]] 
    k_dim_len <- length(k_dim)
    k_aidx <- seq_array_ind(k_dim, col_major = FALSE) 
    
    # the index for the parameters in the whole 
    # sequences of parameters 
    index <- attr(tidx[[k]], "row_major_idx")  

    # number of parameters we could plot for this 
    # particular vector/array parameter 
    k_num_p <- length(index) 

    # number of parameter we would plot
    J <- min(k_num_p, max_width)
    spacing <- 3.5 / max(J, standard_width)

    # the medians for all the kept samples merged 
    sprobs = default_summary_probs()  
    mp <- match(0.5, sprobs) 
    i80p <- match(c(0.1, 0.9), sprobs) 
    med <- summary$quan[index, mp] 
    med <- array(med, dim = c(k_num_p, 1)) 
    i80 <- summary$quan[index, i80p] 
    i80 <- array(i80, dim = c(k_num_p, 2)) 
    rhats <- summary$rhat 
    rhats_cols <- get_rhat_cols(rhats) 
  
    med.chain <- summary$c_quan[index, mp, ]
    med.chain <- array(med.chain, dim = c(k_num_p, sim$chains)) 
    i80.chain <- summary$c_quan[index, i80p, ]
    i80.chain <- array(i80.chain, dim = c(k_num_p, 2, sim$chains))

    rng <- if (display_parallel) range(i80, i80.chain) else range(i80)
    p.rng <- pretty(rng, n = 2)
    b <- height / (max(p.rng) - min(p.rng))
    a <- -(k + height / 2) - b * p.rng[1]
    lines(A + c(0, 0), -k + 0.5 * height * c(-1, 1)) 
    
    # plot a line at zero (if zero is in the range of the mini-plot)
    if (min(p.rng) < 0 & max(p.rng) > 0) {
      lines(A + B * spacing * c(0, J + 1), 
            rep(a, 2), lwd = .5, col = "gray")
    }
    # plot the breaks of the axis
    for (x in p.rng){
      text(A - B * .2, a + b * x, x, cex = cex.axis)
      lines(A + B * c(-.05, 0), rep(a + b * x, 2))
    }
    for (j in 1:J){
      if (display_parallel){
        for (m in 1:chains){
          interval <- a + b * i80.chain[j, , m]

          # When the interval is too tiny, we use the min.width instead
          # of the real one. 
          if (interval[2] - interval[1] < min.width)
            interval <- mean(interval) + c(-.5, .5) * min.width
          segments(x0 = A + B * spacing * (j + .6 *(m - (chains + 1) / 2) / chains), 
                   y0 = interval[1], y1 = interval[2], lwd = .5, 
                   col = chain_cols[(m-1) %% chain_cols.len + 1]) 
        }
      } else {
        lines(A + B * spacing * rep(j, 2), a + b * i80[j,], lwd = .5)
        for (m in 1:chains)
          points(A + B * spacing * j, a + b * med.chain[j, m], 
                 pch = 20, cex = cex.points, 
                 col = chain_cols[(m-1) %% chain_cols.len + 1])
      } 

      # draw an indicator for Rhat
      # (xleft, ybottom, xright, ytop)
       
      if (k_dim_len == 0) 
        rect(A + B * spacing * (j - .5), -k - height / 2 - 0.05 + num_height * .5, 
             A + B * spacing * (j + .5), -k - height / 2 - 0.05 - num_height * .5, col = rhats_cols[j], border = NA) 

      # plot the dimension indexes for this parameter 
      if (k_dim_len  >= 1) { 
        rect(A + B * spacing * (j - .5), -k - height / 2 - 0.05 + num_height * .5, 
             A + B * spacing * (j + .5), -k - height / 2 - 0.05 - num_height * (k_dim_len - .5), col = rhats_cols[j], border = NA) 

        # k_dim: the dimension of parameter k 
        for (m in 1:k_dim_len) {
          index0 <- k_aidx[j, m] 
          if (j == 1)
            text(A+B*spacing*j, -k-height/2-.05-num_height*(m-1), index0, cex=cex_tiny)
          else if (index0 != k_aidx[j - 1, m] & (index0 %% (floor(log10(index0) + 1)) == 0))
            text(A+B*spacing*j, -k-height/2-.05-num_height*(m-1), index0, cex=cex_tiny)

          # Note for `(index0 %% (floor(log10(index0) + 1)) == 0) in the above condition.
          # When 10 <= index0 <= 99, floor(log10(index0) + 1) == 2,
          # so that one index would be drawn out of two consecutive. 
          # That is, we would have 10, 12, 14, 16, etc. 
          # Similarly, when 100 <= index0 <= 999, we draw one out of three
          # though in the case, we do not draw them at all since the max is  
          # 40.  
        }
      }
    } 
    if (J < k_num_p) {
      text (-.015, -k, "*", cex = cex_names, col = alert_col)
      truncated <- TRUE
    } 
  } 
  plot_rhat_legend(0, -n_pars - .5, cex = cex_names)
  if (truncated) {
    text(0, -n_pars - .5 - num_height * 2.5, "*  array truncated for lack of space", 
         adj = c(0, 0), cex = cex_names, col = alert_col)
  } 
  invisible(NULL)
} 

legitimate_model_name <- function(name, obfuscate_name = TRUE) {
  # To make model name be a valid name in C++. 
  # obfuscate_name 

  namep1 <- if (obfuscate_name)  basename(tempfile('model', '')) else 'model' 
  name <- paste(namep1, '_', name, sep = '') 
  gsub('[^[:alnum:]]', '_', name) 
  # return("anon_model")

  # Note: why using different (ideally unique) name?   
  # 
  # The name returned from this function is used 
  # as Rcpp module name and the name for the stan_fit class
  # for each model. Actually we need a unique name. The reason
  # is that it seems if the Rcpp modules have the same name, a newly
  # created model created from compiling the C++ code would replace
  # previous one though the DSO files are different. I guess 
  # that Rcpp implement the module by call the C++ function using .Call, we
  # would always call the function with the same name loaded later. I am 
  # not sure the real reason, but experiments do show that 
  # later modules created would use previous one if the class name
  # in the module is the same. So if obfuscate_name = TRUE, we try
  # to generate a unique name, if FALSE, it is the user's responsibility 
  # to keep the name unique and in the case, users might be able to
  # take advantage of tools such as ccache 
} 

boost_url <- function() {"http://www.boost.org/users/download/"} 

makeconf_path <- function() {
  arch <- .Platform$r_arch
  if (arch == '') 
    return(file.path(R.home(component = 'etc'), 'Makeconf'))
  return(file.path(R.home(component = 'etc'), arch, 'Makeconf'))
} 

is_null_ptr <- function(ns) {
  .Call("is_Null_NS", ns)
}

is_null_cxxfun <- function(cx) {
  # Tell if the returned object from cxxfunction in package inline
  # contains null pointer 
  add <- body(cx@.Data)[[2]]
  # add is of class NativeSymbol
  .Call("is_Null_NS", add)
}

obj_size_str <- function(x) {
  if (x >= 1024^3)       return(paste(round(x/1024^3, 1L), "Gb"))
  else if (x >= 1024^2)  return(paste(round(x/1024^2, 1L), "Mb"))
  else if (x >= 1024)    return(paste(round(x/1024, 1L), "Kb"))
  return(paste(x, "bytes")) 
} 

system_info <- function() {
  paste("OS: ", R.version$system, 
        "; rstan: ",  packageVersion('rstan'), 
        "; Rcpp: ", packageVersion('Rcpp'),
        "; inline: ", packageVersion('inline'), sep = '')
} 

read_comments <- function(file, n) {
  # Read comments beginning with `#`
  # Args:
  #   file: the filename 
  #   n: max number of line; -1 means all 
  .Call("read_comments", file, n, PACKAGE = 'rstan')
} 

sqrfnames_to_dotfnames <- function(fnames) {
  # change names such as alpha[1,1] to alpha.1.1
  gsub('\\]', '', gsub('\\[|,', '.', fnames))
} 


dotfnames_to_sqrfnames <- function(fnames) {
  fnames <- sapply(fnames, 
                   function(i) { 
                     if (!grepl("\\.", i)) return(i)
                     i <- sub("\\.", "[", i)
                     i <- sub("\\s*$", "]", i)
                     i }, USE.NAMES = FALSE)
  gsub("\\.\\s*", ",", fnames)
} 

unique_par <- function(fnames) {
  # obtain parameters from flat names in format of say alpha.1, 
  # alpha.2, beta.1.1, ..., beta.3.4, --- in this case, return
  # c('alpha', 'beta')
  unique(gsub('\\..*', '', fnames)) 
} 


get_dims_from_fnames <- function(fnames, pname) {
  # Get the dimension for a parameter from
  # the flatnames such as "alpha.1.1", ..., "alpha.3.4", the 
  # format of names in the CSV files generated by Stan. 
  # Currently, this function assume fnames are correctly given.  
  # Args:
  #   fnames: a character of names for one (vector/array) parameter
  #   pname: the name for this vector/array parameter such as "alpha"
  #     for the above example 
  
  if (missing(pname)) pname <- gsub('\\..*', '', fnames[1])

  if (length(fnames) == 1 && fnames == pname) 
    return(integer(0)) # a scalar 

  idxs <- sub(pname, '', fnames, fixed = TRUE)
  lp <- gregexpr('\\d+', idxs)

  tfun <- function(name, start, i) {
    last <- attr(start, 'match.length')[i] + start[i] 
    # cat('name=', name, ', start=', start[i], ', last=', last, '.\n', sep = '')
    as.integer(substr(name, start[i], last))
  } 

  dim_len <- length(lp[[1]])
  dims <- integer(dim_len)
  for (i in 1:dim_len) { 
    dimi <- mapply(tfun, idxs, lp, MoreArgs = list(i = i), USE.NAMES = FALSE) 
    dims[i] <- max(dimi) 
  }
  dims
} 

all_int_eq <- function(is) {
  # tell if all integers in 'is' are the same 
  if (!all(is.integer(is)))
    stop("not all are integers")
  min(is) == max(is)
} 
