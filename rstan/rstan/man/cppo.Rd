
\name{cppo}
\alias{get_cppo}
\alias{set_cppo}

\title{Set and get the optimization level and debug flag for compiling the C++ code} 

\description{Set and get the optimization level, which is defined in 
 CXXFLAGS, for compiling the C++ code. Also flag \code{-DDEBUG} 
 or \code{-DNDEBUG} in \code{R_XTRA_CPPFLAGS} is set depending on the
 optimization model.}

\usage{
  set_cppo(mode = c("fast", "presentation2", "presentation1", "debug"))
  get_cppo() 
} 
 
\arguments{
  \item{mode}{A character specifying the optimization level: one of 
              \code{"fast"}, \code{"presentation2"}, \code{"presentation1"}, \code{"debug"},   
              corresponding to optimization level 
              \code{"3"}, \code{"2"}, \code{"1"}, \code{"0"}; defaults to \code{"fast"}.
              The order from \code{"fast"} to \code{"debug"} generally means
              that the generated code runs from the fastest to the slowest. See
              the notes below.}
} 


\value{
  \code{set_cppo} returns the list with element names being  
  \code{CXXFLAGS} and \code{R_XTRA_CPPFLAGS}. Each element
  is the desired flag that is set if the operation is successful; 
  if any problems is encountered, this function will stop and report an error.  

  \code{get_cppo} returns a character string indicating the optimization mode (that is,  
   one of \code{"fast"}, \code{"presentation2"}, \code{"presentation1"}, \code{"debug"}).   
}


\note{
  Since the optimization level is set by using files such as \code{Makevars} (or similar
  file names depending on platforms) in folder 
  \code{.R} under the user's home directory, the side effect is that the optimization
  level set here will be used by \code{R CMD SHLIB} and possibly installing other \R package 
  from source (including installing \pkg{rstan} again). If this is not desired, the 
  created file can be removed. For the same 
  reason, the optimization level that is set stays once \code{set_cppo} is called. If a different
  optimization level is needed, \code{set_cppo} should be called again. 

  Generally for compiling models, we strongly recommend using \code{mode="fast"}. In the 
  case for the purpose of easily debugging code, \code{mode="debug"} can be used. And we recommend 
  calling \code{set_cppo} again to set the optimization mode back to \code{"fast"} after 
  finishing debugging. 
} 

\references{
  The Stan Development Team (2012) 
  \emph{Stan Modeling Language User's Guide and Reference Manual}. 
  \url{http://mc-stan.org/}. 

  help of \code{R CMD SHLIB}. 
} 

\seealso{
  \code{\link{stan}} and \code{\link{stan_model}} 
} 

\examples{\dontrun{
set_cppo("fast") 
get_cppo() 
}} 
