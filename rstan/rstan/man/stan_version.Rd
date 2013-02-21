\name{stan_version}
\alias{stan_version}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Obtain the version of Stan} 
\description{
  The stan version is in form of \code{major.minor.patch};  the
  first version is 1.0.0, indicating major version 1, minor version
  0 and patch level 0.  Functionality only changes with minor versions
  and backward compatibility will only be affected by major versions.
}
\usage{
  stan_version()
}
\value{
  A character string giving the version of Stan used in this version of RStan. 
}
\references{
  The Stan Development Team (\Sexpr[eval=TRUE,results=text,stage=build]{format(Sys.time(),'\%Y')})
  \emph{Stan Modeling Language User's Guide and Reference Manual}. 
  \url{http://mc-stan.org/}. 
}
\seealso{
  \code{\link{stan}} and \code{\link{stan_model}} 
}
\examples{
  stan_version() 
}
\keyword{rstan} 
