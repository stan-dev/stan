\name{stan.model}
\alias{stan.model}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Construct a Stan model} 

\description{
  Construct and \code{stanmodel} object from a model 
  represented by Stan model language. The model 
  is compiled to C++ code if it is not done.  
  The model C++ code plus other auxiliary 
  code is compiled into shared object and then loaded. 
  The loaded module is available for drawing samples.  
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}

\usage{
  stan.model(file, model.name = "anon_model", 
             model.code = "", stanc.ret = NULL, boost.lib = NULL, 
             verbose = FALSE)
}
%- maybe also 'usage' for other objects documented here.

\arguments{
  \item{file}{A character string or a connection that R supports specifying the model 
    in Stan language. 
  }

  \item{model.name}{A character string for naming the model (the
    default is \code{"anon_model"}). 
  }

  \item{model.code}{A character string to specify the model instead of using
    argument file. 
  }
  \item{stanc.ret}{The returned list from \code{stanc}, an alternative way
    to specify the model using previous results.}   
  \item{boost.lib}{The path for boost library \url{http://http://www.boost.org/users/download/}. 
    If specified, this path would be used instead for boost library header files for compiling 
    the model in C++ language.}
  \item{verbose}{\code{TRUE} or \code{FALSE}: indication of whether
    intermediate output inside the function, which might be helpful for debugging
    the model, are printed or not. 
  }

}

\details{
  %%  ~~ If necessary, more details than the description above ~~
  The details of Stan can be found at \url{http://code.googlecode.com/p/stan}. 
}
\value{
  A stanmodel class object (\linkS4class{stanmodel}), which could be
  used later for drawing samples by calling its \code{sampling} function.
  %%  ~Describe the value returned
  %%  If it is a LIST, use
  %%  \item{comp1 }{Description of 'comp1'}
  %%  \item{comp2 }{Description of 'comp2'}
  %% ...
}
\references{
 \url{http://code.google.com/p/stan} 
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
  \code{\linkS4class{stanmodel}} 
  %% ~~objects to See Also as \code{\link{help}}, ~~~
}
\examples{
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{stan}
\keyword{stanmodel}
