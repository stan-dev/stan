\name{makeconf_path}
\alias{makeconf_path}
\title{
Obtain the full path of file \code{Makeconf} 
}


\description{
  Obtain the full path of file \code{Makeconf}, in which, for example
  the flags for compiling C/C++ code are configured.
}

\usage{
makeconf_path() 
} 

\value{
  An character string for the full path of file \code{Makeconf}. 
}

\details{
  The configuration for compiling shared objects using \code{R CMD SHLIB} 
  are set in file \code{Makeconf}. To change how the C++ code is
  compiled, modify this file. For RStan, package \pkg{inline} 
  compiles the C++ code using \code{R CMD SHLIB}. To speed up compiled
  Stan models, increase the optimization level to \code{-O3} defined
  in property \code{CXXFLAGS} in the file \code{Makeconf}.
  This file may also be modified to specify alternative C++ compilers,
  such as clang++ or later versions of g++.

  As of rstan 1.0.2, \code{set_cppo} can be used to create/change 
  a file that can be used to define compiling flags in the user's home
  directoty, in which a flag can be set to overwrite what is 
  set in the system's Makeconf file. 
}

\seealso{
  \code{\link{stan}}, \code{\link{set_cppo}}  
}
\examples{
makeconf_path() 
}
\keyword{rstan} 
