\name{stan.model}
\alias{stan.model}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Construct a Stan model} 

\description{
  A Bayesian statistical model specified in Stan language
  (\url{http://code.google.com/p/stan}) 
  is first compiled to C++ code. The model C++ code plus other auxiliary 
  code is compiled into shared object, which is also loaded then. 
  The loaded module is available for drawing samples.  
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}

\usage{
  stan.model(file, verbose = FALSE, model.name = "anon_model", 
             model.code = "")
}
%- maybe also 'usage' for other objects documented here.

\arguments{
  \item{file}{A character string or a connection that R supports specifying the model 
    in Stan language. 
  }

  \item{verbose}{\code{TRUE} or \code{FALSE}: indication of whether
    intermediate output inside the function, which might be helpful for debugging
    the model, are printed or not. 
  }
  \item{model.name}{A character string for naming the model (the
    default is \code{"anon_model"}). 
  }

  \item{model.code}{A character string to specify the model instead of using
    argument file. 
  }
}

\details{
  %%  ~~ If necessary, more details than the description above ~~
  The details of Stan can be found at \url{http://code.googlecode.com/p/stan}. 
}
\value{
  A stanmodel class object (\linkS4class{stanmodel}), which could be
  then used for drawing samples later. 
  %%  ~Describe the value returned
  %%  If it is a LIST, use
  %%  \item{comp1 }{Description of 'comp1'}
  %%  \item{comp2 }{Description of 'comp2'}
  %% ...
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
  \code{\linkS4class{stanmodel}} 
  %% ~~objects to See Also as \code{\link{help}}, ~~~
}
\examples{
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{stan}
\keyword{stanmodel}
