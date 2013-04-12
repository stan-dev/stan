test_set_cppo <- function() {
  o <- set_cppo('fast', NDEBUG = TRUE)
  l <- get_cppo() 
  checkEquals(l[[1]], 'fast')
  checkEquals(l[[2]], TRUE)
  checkEquals(l[[3]], FALSE)

  o <- set_cppo('debug', NDEBUG = TRUE)
  l <- get_cppo() 
  checkEquals(l[[1]], 'debug')
  checkEquals(l[[2]], FALSE)
  checkEquals(l[[3]], TRUE)

  o <- set_cppo('fast', NDEBUG = FALSE)
  l <- get_cppo() 
  checkEquals(l[[1]], 'fast')
  checkEquals(l[[2]], FALSE)
  checkEquals(l[[3]], FALSE)

  o <- set_cppo('fast', NDEBUG = TRUE)
  l <- get_cppo() 
  checkEquals(l[[1]], 'fast')
  checkEquals(l[[2]], TRUE)
  checkEquals(l[[3]], FALSE)

  o <- set_cppo('presentation2', NDEBUG = TRUE)
  l <- get_cppo() 
  checkEquals(l[[1]], 'presentation2')
  checkEquals(l[[2]], TRUE)
  checkEquals(l[[3]], FALSE)
} 

.tearDown <- function() {
  rstan:::rm_last_makefile()
} 
