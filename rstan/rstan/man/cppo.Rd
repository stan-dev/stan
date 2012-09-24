
\name{cppo}
\alias{get_cppo}
\alias{set_cppo}

\title{Set and get the optimization level for compiling the C++ code} 

\description{Set and get the optimization level, which is defined in 
 CXXFLAGS, for compiling te C++ code}  

\usage{
  set_cppo(level = c("3", "2", "1", "0")) 
  get_cppo() 
} 
 
\arguments{
  \item{level}{A character specifying the optimization level: one of 
    \code{"3", "2", "1", "0"}; defaults to \code{"3"}.} 
} 


\value{
  \code{set_cppo} returns \code{TRUE} if the desired optimization level is set successfully;
  if problems are encountered, it will stop and report an error.  

  \code{get_cppo} returns a character indicating the optimization level. 
}


\note{
  Since the optimization
  level specified in \code{set_cppo} is literally put to \code{CXXFLAGS} that is used 
  for compiling the C++ code when \code{R CMD SHLIB} is called, in general higher optimization 
  level means generating fast code. 

  Since the optimization level is set by using files such as \code{Makevars} in folder 
  \code{.R} under the user's home directory, the side effect is that the optimization
  level set here will be used by \code{R CMD SHLIB} and possibly installing \R package 
  from source. If this is not desired, the created file can be removed to get the 
  default of \R. 
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
set_cppo("3") 
get_cppo() 
}} 
