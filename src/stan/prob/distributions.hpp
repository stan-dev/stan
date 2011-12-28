#ifndef __STAN__PROB__DISTRIBUTIONS_HPP__
#define __STAN__PROB__DISTRIBUTIONS_HPP__

#include "stan/prob/distributions_error_handling.hpp"
#include "stan/prob/distributions_constants.hpp"

#include "stan/prob/distributions/uniform.hpp"
#include "stan/prob/distributions/normal.hpp"
#include "stan/prob/distributions/multi_normal.hpp"
#include "stan/prob/distributions/gamma.hpp"
#include "stan/prob/distributions/inv_gamma.hpp"
#include "stan/prob/distributions_chi_square.hpp"
#include "stan/prob/distributions_inv_chi_square.hpp"
#include "stan/prob/distributions_scaled_inv_chi_square.hpp"
#include "stan/prob/distributions_exponential.hpp"
#include "stan/prob/distributions_wishart.hpp"
#include "stan/prob/distributions_inv_wishart.hpp"
#include "stan/prob/distributions_student_t.hpp"
#include "stan/prob/distributions_beta.hpp"
#include "stan/prob/distributions_dirichlet.hpp"

#include "stan/prob/distributions_cauchy.hpp"
#include "stan/prob/distributions_pareto.hpp"
#include "stan/prob/distributions_double_exponential.hpp"
#include "stan/prob/distributions_weibull.hpp"
#include "stan/prob/distributions_logistic.hpp"
#include "stan/prob/distributions_lognormal.hpp"
#include "stan/prob/distributions_lkj_corr.hpp"
#include "stan/prob/distributions_lkj_cov.hpp"
#include "stan/prob/distributions_bernoulli.hpp"
#include "stan/prob/distributions_categorical.hpp"
#include "stan/prob/distributions_binomial.hpp"
#include "stan/prob/distributions_poisson.hpp"
#include "stan/prob/distributions_neg_binomial.hpp"
#include "stan/prob/distributions_beta_binomial.hpp"
#include "stan/prob/distributions_hypergeometric.hpp"
#include "stan/prob/distributions_multinomial.hpp"

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

