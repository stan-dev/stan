\name{log_prob-methods}
\docType{methods}
\alias{log_prob}
\alias{log_prob-methods}
\alias{log_prob,stanfit-method}
\alias{grad_log_prob}
\alias{grad_log_prob,stanfit-method}
\alias{constrain_pars}
\alias{unconstrain_pars,stanfit-method}
\alias{unconstrain_pars}
\alias{constrain_pars,stanfit-method}
\alias{get_num_upars}
\alias{get_num_upars,stanfit-method}

\title{model's \code{log_prob} and \code{grad_log_prob} functions} 

\description{
  Using model's \code{log_prob} and \code{grad_log_prob} functions  
  on the unconstrained space of model parameters. So sometimes we   
  need convert the values of parameters from their support defined 
  in parameter block (which might be constrained, and for simplicity, 
  we call it constrained space) to unconstrained space and vice versa. 
  \code{constrained_pars} and \code{unconstrain_pars} can be used then.
} 
\usage{
  %% log_prob(object, upars) 
  \S4method{log_prob}{stanfit}(object, upars)
  %% grad_log_prob(object, upars) 
  \S4method{grad_log_prob}{stanfit}(object, upars)
  %% get_num_upars(object)
  \S4method{get_num_upars}{stanfit}(object)
  %% constrain_pars(object, upars)
  \S4method{constrain_pars}{stanfit}(object, upars)
  %% unconstrain_pars(object, pars)
  \S4method{unconstrain_pars}{stanfit}(object, pars)
} 

\section{Methods}{
  \describe{
    \item{log_prob}{\code{signature(object = "stanfit")}}{Compute the log posterior 
      (\code{lp__}) for the model represented by a \code{stanfit} object. 
    }
    \item{grad_log_prob}{\code{signature(object = "stanfit")}}{Compute the gradients
      for \code{log_prob} as well as the log posterior. The latter is returned as 
      an attribute. 
    }
    \item{get_num_upars}{\code{signature(object = "stanfit")}}{Get the number
      of unconstrained parameters.
    } 
    \item{constrain_pars}{\code{signature(object = "stanfit")}}{Convert values
      of the parameter from unconstrained space (given as a vector) to their
      constrained space (returned as a named list).}
    \item{unconstrain_pars}{\code{signature(object = "stanfit")}}{Contrary to
      \code{constrained}, conert values of the parameters from constrained
      to unconstrained space.
    }
  }
} 

\arguments{
  \item{object}{An object of class \code{\linkS4class{stanfit}}.}
  \item{pars}{An list specifying the values for all parameters on the
    constrained space.} 
  \item{upars}{An vector for specifying the values for all parameters 
    on the unconstrained space.}  
} 

\value{
  \code{log_prob} returns a value (up to an additive constant) the log posterior. 

  \code{grad_log_prob} returns a vector of the gradients.  Additionaly, the vector
  has an attribute named \code{log_prob} being the value the same as \code{log_prob}
  is called for the input parameters. 

  \code{get_num_upars} returns the number of parameters on the unconstrained space. 

  \code{constrain_pars} returns a list and \code{unconstrain_pars} returns a vector. 
}

\references{
  The Stan Development Team (\Sexpr[eval=TRUE,results=text,stage=build]{format(Sys.time(),'\%Y')})
  \emph{Stan Modeling Language User's Guide and Reference Manual}. 
  \url{http://mc-stan.org}. 
}

\seealso{
  \code{\linkS4class{stanfit}} 
} 

\examples{\dontrun{
# see the examples in the help for stanfit as well
# do a simple optimization problem 
opcode <- "
parameters {
  real y;
}
model {
  lp__ <- log(square(y - 5) + 1);
}
"
tfun <- function(y) log_prob(opfit, y)
tgrfun <- function(y) grad_log_prob(opfit, y)
or <- optim(1, tfun, tgrfun, method = 'BFGS')
print(or)

# return the gradient as an attribute
tfun2 <- function(y) { 
  g <- grad_log_prob(opfit, y) 
  lp <- attr(g, "log_prob")
  attr(lp, "gradient") <- g
  lp
} 

or2 <- nlm(tfun2, 10)
or2 
}} 
