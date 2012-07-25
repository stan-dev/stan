\name{stan.dump}
\alias{stan.dump}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Dump the data for a Stan model to R dump file
that Stan supports
%%  ~~function to do ... ~~
}


\description{
  This function takes a vector of names of \R objects and produces
  text representations of the objects on a file or connection.
  A \code{stan.dump} can usually be used as data input of the 
  Stan package (\url{http://stan.googlecode.com}) or 
  \code{\link{source}}d into another \R session.
}

\usage{
stan.dump(list, file, append = FALSE, 
          envir = parent.frame(),
          width = options("width")$width)
}

\arguments{
  \item{list}{character.  The names of one or more \R objects to be dumped.}

  \item{file}{either a character string naming a file or a
    \link{connection}.  \code{""} indicates output to the console.}

  \item{append}{if \code{TRUE} and \code{file} is a character string,
    output will be appended to \code{file}; otherwise, it will overwrite
    the contents of \code{file}.}

  \item{envir}{the environment to search for objects.}
  \item{width}{the width for maximum characters on a line.  
    The output is broken into lines with \code{width.}}
}

\details{
%%  ~~ If necessary, more details than the description above ~~
}

\value{
  An invisible character vector containing the names of the objects
  which were dumped.
}
\references{
%% ~put references to the literature/web site here ~
}
\author{
%%  ~~who you are~~
}
\note{
  \code{stan.dump} only dumps numeric data, which could be 
  scale, vector, array, or matrix.  
%%  ~~further notes~~
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
\code{\link{dump}} 
%% ~~objects to See Also as \code{\link{help}}, ~~~
}
\examples{
##---- Should be DIRECTLY executable !! ----
##-- ==>  Define data, use random,
##--	or do  help(data=index)  for the standard data sets.

## The function is currently defined as
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{rstan} 
\keyword{dump}  
