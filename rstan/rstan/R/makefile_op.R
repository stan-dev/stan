#############################################################
###  deal with the optimization level for c++ compilation 

get_makefile_txt <- function() { 
  # get the all makefile content used for R CMD SHLIB 
  # The order or files to look for are following the code 
  # of function .SHLIB in packages tools/R/install.R 
  # 
  # Return: 
  #   a character vector in which each element is a line of 
  #   the makefile 
  WINDOWS <- .Platform$OS.type == "windows"
  rarch <- Sys.getenv("R_ARCH") # unix only
  if (WINDOWS && nzchar(.Platform$r_arch))
    rarch <- paste0("/", .Platform$r_arch)
  site <- file.path(R.home("etc"), rarch, "Makevars.site")

  sys_makeconf <- file.path(R.home("etc"), rarch, "Makeconf")
  if (!file.exists(sys_makeconf)) 
    stop(paste("file ", sys_makeconf, " does not exist", sep = '')) 
  makefiles <-
    c(sys_makeconf, 
      if (file.exists(site)) site,
      file.path(R.home("share"), "make", if (WINDOWS) "winshlib.mk" else "shlib.mk"))

  if (WINDOWS) { 
    if (rarch == "/x64" && 
        file.exists(f <- path.expand("~/.R/Makevars.win64"))) 
      makefiles <- c(makefiles, f) 
    else if (file.exists(f <- path.expand("~/.R/Makevars.win"))) 
      makefiles <- c(makefiles, f) 
    else if (file.exists(f <- path.expand("~/.R/Makevars"))) 
      makefiles <- c(makefiles, f) 
  } else { 
    if (file.exists(f <- path.expand(paste("~/.R/Makevars", 
                                            Sys.getenv("R_PLATFORM"), 
                                            sep = "-")))) 
       makefiles <- c(makefiles, f) 
    else if (file.exists(f <- path.expand("~/.R/Makevars"))) 
       makefiles <- c(makefiles, f) 
  }

  do.call(c, lapply(makefiles, function(f) readLines(f, warn = FALSE))) 
} 

get_makefile_flags <- function(FLAGNAME, makefile_txt, headtotail = FALSE) { 
  # get current CXXFLAGS used for in R CMD SHLIB, which controls
  # how the generated C++ code for stan model are compiled 
  # at which optimization level. 
  # Args: 
  #   FLAGNAME: the name of the flag of interest such as CXXFLAGS 
  #   makefile_txt: a character vector each element of which is a line
  #     of the make file, typically obtained by get_makefile_txt() 
  # Return: 
  #   the whole line that defines CXXFLAGS 

  if (missing(makefile_txt)) 
    makefile_txt <- get_makefile_txt() 

  lineno <- -1 
  iseq <- if (headtotail) 1:length(makefile_txt) else length(makefile_txt):1 
  for (i in iseq) { 
    pattern <- paste("^\\s*", FLAGNAME, "\\s*=", sep = '')
    if (grepl(pattern, makefile_txt[i])) {
      lineno <- i 
      break;
    }
  } 

  if (-1 != lineno) return(makefile_txt[lineno]) 
  paste(FLAGNAME, " = ", sep = '') 
} 

last_makefile <- function() {
  # Return the last makefile for 'R CMD SHLIB'. 
  # In essential, R CMD SHLIB uses GNU make.  
  # R CMD SHLIB uses a series of files as the whole makefile.  
  # This function return the last one, where we can set flags 
  # to overwrite what is set before. 
  # 
  WINDOWS <- .Platform$OS.type == "windows"
  rarch <- Sys.getenv("R_ARCH") # unix only
  if (WINDOWS && nzchar(.Platform$r_arch))
    rarch <- paste0("/", .Platform$r_arch)

  if (WINDOWS) { 
    if (rarch == "/x64") {
       f <- path.expand("~/.R/Makevars.win64") 
       if (file.exists(f)) return(f) 
    }
    f <- path.expand("~/.R/Makevars.win") 
    if (file.exists(f)) return(f) 
    return(path.expand("~/.R/Makevars"))
  }
  f <- path.expand(paste("~/.R/Makevars", Sys.getenv("R_PLATFORM"), sep = "-"))
  if (file.exists(f)) return(f) 
  return(path.expand("~/.R/Makevars"))
} 

set_makefile_flags <- function(flags) { 
  # Set user-defined CXXFLAGS for R CMD SHLIB by 
  # set CXXFLAGS or others in the last makefile obtained by last_makefile 
  # Args: 
  #   flags: a named list with names CXXFLAGS and R_XTRA_CPPFLAGS 
  # 
  # Return: 
  #   TRUE if it is successful, otherwise, the function
  #   stops and reports an error. 
  # 
  lmf <- last_makefile() 
  homedotR <- dirname(lmf) 
  if (!file.exists(homedotR)) {
    if (!dir.create(homedotR, showWarnings = FALSE, recursive = TRUE))
      stop(paste("failed to create directory ", homedotR, sep = '')) 
  } 

  flagnames <- names(flags) 
  paste_bn <- function(...)  paste(..., sep = '\n') 
  flags <- lapply(flags, function(x) paste(x, " #set_by_rstan")) 

  # the Makevars file does not exist 
  if (!file.exists(lmf)) {
    if (file.access(homedotR, mode = 2) < 0 || 
        file.access(homedotR, mode = 1) < 0 ) 
      stop(paste("directory ", homedotR, " is not writable", sep = '')) 
    cat(paste("# created by rstan at ", date(), sep = ''), '\n', file = lmf) 
    cat(do.call(paste_bn, flags), file = lmf, append = TRUE) 
    return(invisible(NULL)) 
  } 

  if (file.access(lmf, mode = 2) < 0) 
    stop(paste(lmf, " is not writable", sep = '')) 

  lmf_txt <- readLines(lmf, warn = FALSE) 
  if (length(lmf_txt) < 1) {  # empty file 
    cat(do.call(paste_bn, flags), file = lmf, append = TRUE) 
    return(invisible(NULL))  
  } 

  # change the existing file 
  makefile_exta <- "" 
  for (fname in flagnames) { 
    found <- FALSE 
    for (i in length(lmf_txt):1) {
      pattern <- paste("^\\s*", fname, "\\s*=", sep = '')
      if (grepl(pattern, lmf_txt[i])) { 
        lmf_txt[i] <- flags[[fname]] 
        found <- TRUE 
        break
      }
    }
    if (!found)
      makefile_exta <- paste(makefile_exta, flags[[fname]], sep = '\n') 
  } 

  cat(paste(lmf_txt, collapse = '\n'), file = lmf) 
  cat(makefile_exta, file = lmf, append = TRUE) 
  invisible(NULL) 
} 

rm_last_makefile <- function(force = TRUE) {
  # remove the file given by last_makefile()
  # return: 
  #  ‘0’ for success, ‘1’ for failure, invisibly. 
  #  '-1' file not exists 
  lmf <- last_makefile() 
  msgs <- c(paste(lmf, " does not exist", sep = ''), 
            paste("removed file ", lmf, sep = ''), 
            paste("failed to remove file ", lmf, sep = '')) 
  ret <- -1 
  if (file.exists(lmf)) ret <- unlink(lmf, force = force) 
  cat(msgs[ret + 2], '\n') 
  invisible(ret) 
} 

get_cxxo_level <- function(str) {
  # obtain the optimization level from a string like -O3 
  # Return: 
  #   the optimization level after -O as a character 
  # 
  p <- regexpr('-O.', str, perl = TRUE) + 2 
  if (p == 1) return("")  # not found 
  substr(str, p, p)
} 

if_debug_defined <- function(str) {
  grepl("DDEBUG", str)
} 

if_ndebug_defined <- function(str) {
  grepl("DNDEBUG", str)
} 


rm_rstan_makefile_flags <- function() {
  # remove flags in $HOME/.R/Makevars with #rstan 
  lmf <- last_makefile() 
  if (!file.exists(lmf)) return(NULL)
  if (file.access(lmf, mode = 2) < 0) 
    stop(paste(lmf, " is not writable", sep = '')) 
  lmf_txt <- readLines(lmf, warn = FALSE) 
  if (length(lmf_txt) < 1) return(NULL)
  for (i in length(lmf_txt):1) {
    if (grepl("#set_by_rstan", lmf_txt[i])) 
      lmf_txt[i] <- NA  
  } 
  lmf_txt <- lmf_txt[!is.na(lmf_txt)] 
  cat(paste(lmf_txt, collapse = '\n'), file = lmf) 
  message("compiler flags set by rstan are removed") 
  invisible(NULL)
} 

set_cppo <- function(mode = c("fast", "presentation2", "presentation1", "debug")) { 
  # set the optimization level for compiling C++ code using R CMD SHLIB
  # Args:
  #   mode: one of fast, presentation2, presentation1, debug, corresponding
  #   to level 3, 2, 1, 0. 
  #   In addition, when debug mode is specified, it will add -g to CXXFLAGS. 
  #   In mode other than debug, -g is removed if exists. 
  # Return: 
  #   A list with names CXXFLAGS and R_XTRA_CPPFLAGS that are set 
  # 
  mode = match.arg(mode) 
  level = c(3, 2, 1, 0)[match(mode, c("fast", "presentation2", "presentation1", "debug"))]  
  makefile_txt <- get_makefile_txt() 

  # if (mode == 'system') {
  #   rm_rstan_makefile_flags() 
  #   message("compiler flags set by rstan are removed") 
  #   retrun(invisible(list()))
  # } 

  sys_cxxflags <- get_makefile_flags("CXXFLAGS", makefile_txt, headtotail = TRUE)
  curr_cxxflags <- get_makefile_flags("CXXFLAGS", makefile_txt) 
  curr_r_xtra_cppflags <- get_makefile_flags("R_XTRA_CPPFLAGS", makefile_txt) 
  if (is.numeric(level)) 
    level <- as.character(level) 
  if (!level %in% c('0', '1', '2', '3', 's')) 
    stop(paste(level, " might not be a legal optimization level for C++ compilation", sep = '')) 
  old_olevel <- get_cxxo_level(curr_cxxflags) 
  if (old_olevel == level)  {
    if ((level == '0' && grepl("-DDEBUG", curr_r_xtra_cppflags) && grepl(" -g", curr_cxxflags)) ||  
        (level != '0' && grepl("-DNDEBUG", curr_r_xtra_cppflags) && !grepl(" -g", curr_cxxflags))) 
    message(paste('mode ', mode, ' for compiling C++ code has been set already', sep = ''))
    return(invisible(list(CXXFLAGS = curr_cxxflags, 
                          R_XTRA_CPPFLAGS = curr_r_xtra_cppflags))) 
  } 
  new_cxxflags <- if (old_olevel == '')  { 
                   paste(curr_cxxflags, " -O", level, sep = '') 
                 } else {
                   sub(paste("-O", old_olevel, sep = ''), 
                       paste("-O", level, sep = ''), curr_cxxflags, fixed = TRUE) 
                 } 
  if (level == '0') { 
    if (!grepl(" -g", new_cxxflags)) new_cxxflags <- sub("\\s*$", " -g", new_cxxflags) 
    new_r_xtra_cppflags <- sub("-DNDEBUG", "-DDEBUG", curr_r_xtra_cppflags, fixed = TRUE) 
  } else {
    new_cxxflags <- sub("-g", "", new_cxxflags, fixed = TRUE)
    new_r_xtra_cppflags <- sub("-DDEBUG", "-DNDEBUG", curr_r_xtra_cppflags, fixed = TRUE) 
  } 
  flags <- list(CXXFLAGS = new_cxxflags,
                R_XTRA_CPPFLAGS = new_r_xtra_cppflags) 
  set_makefile_flags(flags) 
  message(paste('mode ', mode, ' for compiling C++ code is set', sep = ''))
  invisible(flags) 
} 

get_cppo <- function() {
  # get the optimization level as a character 
  curr_cxxflags <- get_makefile_flags("CXXFLAGS", get_makefile_txt())
  l <- get_cxxo_level(curr_cxxflags)  
  if ("" == l) l <- "0" 
  p <- match(l, c("3", "2", "1", "0")) 
  c("fast", "presentation2", "presentation1", "debug")[p]
} 
