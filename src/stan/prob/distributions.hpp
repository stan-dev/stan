#ifndef __STAN__PROB__DISTRIBUTIONS_HPP__
#define __STAN__PROB__DISTRIBUTIONS_HPP__

#include "stan/prob/error_handling.hpp"
#include "stan/prob/constants.hpp"

#include "stan/prob/distributions/uniform.hpp"
#include "stan/prob/distributions/normal.hpp"
#include "stan/prob/distributions/multi_normal.hpp"
#include "stan/prob/distributions/gamma.hpp"
#include "stan/prob/distributions/inv_gamma.hpp"
#include "stan/prob/distributions/chi_square.hpp"
#include "stan/prob/distributions/inv_chi_square.hpp"
#include "stan/prob/distributions/scaled_inv_chi_square.hpp"
#include "stan/prob/distributions/exponential.hpp"
#include "stan/prob/distributions/wishart.hpp"
#include "stan/prob/distributions/inv_wishart.hpp"
#include "stan/prob/distributions/student_t.hpp"
#include "stan/prob/distributions/beta.hpp"
#include "stan/prob/distributions/dirichlet.hpp"

#include "stan/prob/distributions/cauchy.hpp"
#include "stan/prob/distributions/pareto.hpp"
#include "stan/prob/distributions/double_exponential.hpp"
#include "stan/prob/distributions/weibull.hpp"
#include "stan/prob/distributions/logistic.hpp"
#include "stan/prob/distributions/lognormal.hpp"
#include "stan/prob/distributions/lkj_corr.hpp"
#include "stan/prob/distributions/lkj_cov.hpp"
#include "stan/prob/distributions/bernoulli.hpp"
#include "stan/prob/distributions/categorical.hpp"
#include "stan/prob/distributions/binomial.hpp"
#include "stan/prob/distributions/poisson.hpp"
#include "stan/prob/distributions/neg_binomial.hpp"
#include "stan/prob/distributions/beta_binomial.hpp"
#include "stan/prob/distributions/hypergeometric.hpp"
#include "stan/prob/distributions/multinomial.hpp"

namespace stan {
  namespace prob {

    using Eigen::Array;
    using Eigen::Matrix;
    using Eigen::DiagonalMatrix;
    using Eigen::Dynamic;
    using namespace std;
    using namespace stan::maths;

    // UNIVARIATE CUMULATIVE DISTRIBUTIONS




    // CONTINUOUS, MULTIVARIATE

    

    // DISCRETE, UNIVARIATE MASS FUNCTIONS


    // DISCRETE, MULTIVARIATE MASS FUNCTIONS

    // LINEAR SCALE DENSITIES AND MASS FUNCTIONS
  
    double uniform(double y, double alpha, double beta) {
      return 1.0/(beta - alpha);
    }

    double normal(double y, double loc, double scale) {
      return exp(normal_log(y,loc,scale));
    }

    double gamma(double y, double alpha, double beta) {
      return exp(gamma_log(y,alpha,beta));
    }

    double chi_square(double y, double dof) {
      return exp(chi_square_log(y,dof));
    }

    double inv_chi_square(double y, double dof) {
      return exp(inv_chi_square_log(y,dof));
    }

    double scaled_inv_chi_square(double y, double nu, double s) {
      return exp(scaled_inv_chi_square_log(y,nu,s));
    }

    double exponential(double y, double beta) {
      return exp(exponential_log(y,beta));
    }

  }
}

#endif

