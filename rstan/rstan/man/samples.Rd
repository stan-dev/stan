\name{samples-methods}
\docType{methods}
\alias{samples} 
\alias{samples-methods}
\alias{samples,stanmodel-method}

\title{\code{samples}: draws samples from Stan model}
\description{
  Draw samples from the model defined by class \code{stanmodel}. 
}

\usage{
  %% samples(object, \dots)  
  \S4method{samples}{stanmodel}(object, data, n.chains = 1L, n.iter = 2000L, 
    n.warmup = as.integer(n.iter / 2), 
    thin = 1L, init.t = 'random', init.v = NULL, seed, 
    sample.file, \dots, verbose = FALSE)   
} 

\section{Methods}{
  \describe{
    \item{samples}{\code{signature(object = "stanmodel")}}{
      %%  ~~describe this method here~~
      Call NUTS to draw samples from the model defined by class \code{"stanmodel"} 
      given the data, init values, etc. 
    }
  }
}

\arguments{
  \item{object}{An object of class \code{"\linkS4class{stanmodel}"} object.}

  \item{data}{An object of class \code{"list"} or \code{"environment"}
    providing the data for the model.}    

 \item{n.chains}{A positive integer specifying number of chains. The default is 1. (Not 
   implemented now, that is, it is always 1.)} 

 \item{n.iter}{A positive integer specifying how many iterations for all the
   chains (including warmup). The default is 2000.}

 \item{n.warmup}{A positive integer specifying number of initial samples to be
                 discarded. The default is \code{"n.iter / 2"}.}

 \item{thin}{A positive integer specifying the period between saved samples
   after warmup. The default is \code{"max(1, floor(n.iter - n.warmup) / 1000)"}.}

 \item{init.t}{One of \code{0} (or \code{"0"}), \code{"user"}, or \code{"random"} 
   indicating how the initial values of parameters are specified. \code{"0"}: 
   initialize all to be zero (FIXME: check if is on the unconstrained support); 
   \code{"user"}: specify by the user in \code{"init.v"}; and 
   \code{"random"}: randomly generated.  }

 \item{init.v}{An object of class \code{"list"}, a named list for specifying
   initial values for the model parameters. Note that now the user must specify 
   them for all parameters.}

 \item{seed}{An integer for random number generation seed. 
   The default is randomly generated from time.} 

 \item{sample.file}{A character string of file name for specifying where the 
   samples are written into.} 

 \item{\dots}{Not implemented now.  They are supposed to provide 
   additional parameters for NUTS.}
   
 \item{verbose}{\code{TRUE} or \code{FALSE}: indication of whether
   intermediate output inside the function, which might be helpful for
   debugging the model, are printed or not.}
      
}
\value{
  \code{TRUE} indicates success of sampling; otherwise \code{FALSE}. 
} 

\examples{
\dontrun{
}
} 

\keyword{methods}
\keyword{stanmodel}
% \keyword{ ~~ other possible keyword(s) ~~ }
