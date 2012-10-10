
\name{stan_demo}
\alias{stan_demo} 

\title{Demonstrate examples included in Stan} 

\description{Stan includes a variety of examples and most of the BUGS example models 
  that are translated into Stan modeling language. One example is chosen from a list
  created from matching user input and gets fitted in the demonstration.} 
 
\usage{
  stan_demo(model = NULL, ...) 
} 
 
\arguments{
  \item{model}{A character string for model name to specify which model
    will be used for demonstration.  The default is \code{NULL}, where a list
    of all available models are printed out for selection.} 
  \item{\dots}{passed to \code{stan}.}
} 


\value{
  An S4 object of \code{stanfit} returned from \code{stan}. 
}

\references{
  The Stan Development Team (2012) 
  \emph{Stan Modeling Language User's Guide and Reference Manual}. 
  \url{http://mc-stan.org/}. 
} 

\seealso{
  \code{\link{stan}} 
} 

\examples{\dontrun{
 stan_demo()  # select from all available examples 
 stan_demo("dogs") # run a specific example 
}} 
