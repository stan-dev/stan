\name{stan_demo}
\alias{stan_demo} 

\title{Demonstrate examples included in Stan} 

\description{Stan includes a variety of examples and most of the BUGS example models 
  that are translated into Stan modeling language. One example is chosen from a list
  created from matching user input and gets fitted in the demonstration.} 
 
\usage{
  stan_demo(model = character(0), ...) 
} 
 
\arguments{
  \item{model}{A character string for model name to specify which model
    will be used for demonstration.  The default is an empty string, which
    prompts the user to select one the available models. If \code{model = 0},
    a character vector of all models is returned without any user intervention.
    If \code{model = i} where \code{i > 0}, then the ith available model is
    chosen without user intervention, which is useful for testing.}
  \item{\dots}{passed to \code{stan}.}
} 


\value{
  An S4 object of \code{stanfit} returned from \code{stan}, unless
  \code{model = 0}, in which case a character vector of paths to
  available models is returned.
}

\references{
  The Stan Development Team (\Sexpr[eval=TRUE,results=text,stage=build]{format(Sys.time(),'\%Y')})
  \emph{Stan Modeling Language User's Guide and Reference Manual}. 
  \url{http://mc-stan.org/}. 
} 

\seealso{
  \code{\link{stan}} 
} 

\examples{
  model_names <- gsub(".stan$", "", basename(stan_demo(0)))
  print(model_names)
  \dontrun{
     dogsfit <- stan_demo("dogs") # run the dogs model
     fit1 <- stan_demo(1) # run model_names[1]
  }
} 
