\name{stan_rdump}
\alias{stan_rdump}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Dump the data for a Stan model to \R dump file
in the limited format that Stan can read.
%%  ~~function to do ... ~~
}


\description{
  This function takes a vector of names of \R objects and outputs
  text representations of the objects to a file or connection.
  The file created by \code{stan_rdump} is typically used as data 
  input of the Stan package (\url{http://mc-stan.org/}) or 
  \code{\link{source}}d into another \R session.
}

\usage{
stan_rdump(list, file = "", append = FALSE, 
          envir = parent.frame(),
          width = options("width")$width)
}

\arguments{
  \item{list}{A character string: the names of one or more \R objects to be
    dumped. See the note below.}

  \item{file}{Either a character string naming a file or a
    \link{connection}.  \code{""} indicates output to the console.}

  \item{append}{Logical: if \code{TRUE} and \code{file} is a character string,
    output will be appended to \code{file}; otherwise, it will overwrite
    the contents of \code{file}.}

  \item{envir}{The environment to search for objects.}
  \item{width}{The width for maximum characters on a line.  
    The output is broken into lines with \code{width.}}
}

\value{
  An invisible character vector containing the names of the objects
  that were dumped.
}
\note{
  \code{stan_rdump} only dumps numeric data, which can be 
  a scalar, vector, matrix, or (multidimensional) array.
  In the case if it is a \code{data.frame}, function 
  \code{data.matrix} is applied to the data frame before
  dumping. 

  \code{stan_rdump} will check whether the names of objects
  are legal variable names in Stan. If an illegal name is 
  found, data will be dumped with a warning. However, passing the 
  name checking does not necessarily mean that the name is 
  legal. More details regarding rules of variable names in Stan can
  be found in Stan's manual. 
  
  If objects with specified names are not found, a warning will
  be issued. 
}

\references{
  Stan Development Team (2012) 
  \emph{Stan Modeling Language User's Guide and Reference Manual}. 
  \url{http://mc-stan.org}. 
}

\seealso{
  \code{\link{dump}} 
}
\examples{
# set variables in global environment
a <- 17.5;
b <- c(1,2,3);
# write variables a and b to file ab.data.R in working directory
stan_rdump(list('a','b'), "ab.data.R")

x <- 1; y <- 1:10; z <- array(1:10, dim = c(2,5)) 
stan_rdump(ls(pattern = '^[xyz]'), "xyz.Rdump")
cat(paste(readLines("xyz.Rdump"), collapse = '\n'), '\n')
unlink("xyz.Rdump")
}
\keyword{rstan} 
