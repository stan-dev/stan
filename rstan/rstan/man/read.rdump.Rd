\name{read.rdump}
\alias{read.rdump}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Read data in an R dump file to a list}
\description{Create an R list from a R dump file}
\usage{
read.rdump(f)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{f}{A character string providing the dump file name.}
}
\details{
The R dump file can be read directly by R function \code{source}, which
by default would read the data into the user's workspace (the global environment). 
This function instead read the data to a list, making it convenient to 
prepare data for function \code{stan}. 
}
\value{
A list containing all the data defined in the dump file.
}
\seealso{
\code{\link{stan.rdump}}; \code{\link{dump}} 
%% ~~objects to See Also as \code{\link{help}}, ~~~
}
\examples{
x <- 1; y <- 1:10; z <- array(1:10, dim = c(2,5)) 
stan.rdump(ls(pattern = '^[xyz]'), "xyz.Rdump")
l <- read.rdump('xyz.Rdump')
print(l) 
}
\keyword{rstan} 
