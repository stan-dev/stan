\name{makeconf.path}
\alias{makeconf.path}
\title{
Obtain the full path of file \code{Makeconf} 
}


\description{
  Obtain the full path of file \code{Makeconf}, in which, for example
  the flags for compiling C/C++ code are configured.
}

\usage{
makeconf.path() 
} 

\value{
  An character string for the full path of file \code{Makeconf}. 
}

\details{
  The configuration for compiling shared objects using \code{R CMD SHLIB} 
  are set in file \code{Makeconf}. To change how the C/C++ code are 
  compiled, we can modify this file. In RStan, package \pkg{inline} 
  compiles the C++ code through using \code{R CMD SHLIB}. To make the 
  code run fast, we prefer to using the highest optimization level. 
  In compilers such as gcc (\url{http://gcc.gnu.org}), we would like to set
  \code{-O3} in \code{CXXFLAGS}.  In addition, we can also modify this file
  to use other C++ compilers. 
}

\seealso{
  \code{\link{stan}} 
}
\examples{
makeconf.path() 
}
\keyword{rstan} 
