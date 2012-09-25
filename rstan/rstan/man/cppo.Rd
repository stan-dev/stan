
\name{cppo}
\alias{get_cppo}
\alias{set_cppo}

\title{Set and get the optimization level and debug flag for compiling the C++ code} 

\description{Set and get the optimization level, defined in 
 CXXFLAGS, for compiling the C++ code. Also flag \code{-DDEBUG} 
 or \code{-DNDEBUG} in \code{R_XTRA_CPPFLAGS} is set depending on the
 optimization mode.  
}

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
  Since the optimization level is set by using a file with name \code{Makevars} (or 
  a similar file with names such as \code{Makevars.win64}, \code{Makevars.win}, 
  \code{Makevars-$R_PLATFROM} depending on platforms) in folder 
  \code{.R} under the user's home directory, the \emph{side effect} is that the optimization
  level set here will be used by \code{R CMD SHLIB} and possibly installing other \R package 
  from source (including installing \pkg{rstan} again). If this is not desired, the 
  created file can be removed. For the same 
  reason, the optimization level that is set stays once \code{set_cppo} is called. If a different
  optimization level is needed, \code{set_cppo} needs to be called again. 

  Generally for compiling models, what is strongly recommended is to use \code{mode="fast"}. When
  debugging C++ code using for example \code{gdb}, \code{mode="debug"} can be used, in which
  case \code{-g} flag is also set in \code{CXXFLAGS}. And we recommend 
  calling \code{set_cppo} again to set the optimization mode back to \code{"fast"} after 
  finishing debugging. 
} 

\references{
  The Stan Development Team (2012) 
  \emph{Stan Modeling Language User's Guide and Reference Manual}. 
  \url{http://mc-stan.org/}. 

  help of \code{R CMD SHLIB}. 

  manual of \emph{gcc} and \emph{clang}. 
} 

\seealso{
  \code{\link{stan}} and \code{\link{stan_model}} 
} 

\examples{\dontrun{
set_cppo("fast") 
get_cppo() 
}} 
