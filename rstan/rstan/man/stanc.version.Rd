\name{stanc.version}
\alias{stanc.version}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Obtain the version of stanc in Stan} 
\description{
  The stan version is in form of MAJOR_VERION.MINOR_VERSION.PATCH_VERSION (the
  first version is 1.0.0.). 
}
\usage{
  stanc.version()
}
\value{
  A character string giving the version of stanc. 
}
\references{
  The Stan Development Team (2012) 
  \emph{Stan Modeling Language: User's Guide and Reference}. 
  \url{http://mc-stan.org}. 
}
\seealso{
  \code{\link{stan}} and \code{\link{stan.model}} 
}
\examples{
  stanc.version() 
}
\keyword{rstan} 
