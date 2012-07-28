\name{rstan.options}
\alias{rstan.options}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Set and read options used in RStan
%%  ~~function to do ... ~~
}
\description{
Set and read options used in RStan. Some arguments are programmed as options
so that can be controlled by user. 
}
\usage{
rstan.options(...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{\dots}{Arguments must be in forms of \code{argument = value}.
  Then option \code{argument} is set as \code{value}. In the case
  of just obtaining options, each argument could be a character string.}
}
\details{
%%  ~~ If necessary, more details than the description above ~~
}
\value{
A list of the old options if exists. If an option does
not exist before, it is \code{NA}. When there is only 
one option, the old value is returned. 
}
\references{
%% ~put references to the literature/web site here ~
}
\author{
%%  ~~who you are~~
}
\note{
%%  ~~further notes~~
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
}
\examples{}
\keyword{rstan} 
