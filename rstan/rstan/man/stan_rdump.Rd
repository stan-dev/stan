\name{stan.rdump}
\alias{stan.rdump}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Dump the data for a Stan model to \R dump file
that Stan supports
%%  ~~function to do ... ~~
}


\description{
  This function takes a vector of names of \R objects and outputs
  text representations of the objects to a file or connection.
  The file created by \code{stan.rdump} is typically used as data 
  input of the Stan package (\url{http://mc-stan.org}) or 
  \code{\link{source}}d into another \R session.
}

\usage{
stan.rdump(list, file, append = FALSE, 
          envir = parent.frame(),
          width = options("width")$width)
}

\arguments{
  \item{list}{character. The names of one or more \R objects to be dumped.}

  \item{file}{either a character string naming a file or a
    \link{connection}.  \code{""} indicates output to the console.}

  \item{append}{if \code{TRUE} and \code{file} is a character string,
    output will be appended to \code{file}; otherwise, it will overwrite
    the contents of \code{file}.}

  \item{envir}{the environment to search for objects.}
  \item{width}{the width for maximum characters on a line.  
    The output is broken into lines with \code{width.}}
}

\value{
  An invisible character vector containing the names of the objects
  that were dumped.
}
\note{
  \code{stan.rdump} only dumps numeric data, which can be 
  scalar, vector, array, or matrix.  
}

\seealso{
  \code{\link{dump}} 
}
\examples{
x <- 1; y <- 1:10; z <- array(1:10, dim = c(2,5)) 
stan.rdump(ls(pattern = '^[xyz]'), "xyz.Rdump")
cat(paste(readLines("xyz.Rdump"), collapse = '\n'), '\n')
unlink("xyz.Rdump")
}
\keyword{rstan} 
