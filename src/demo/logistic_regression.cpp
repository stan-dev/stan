#include "stan/agrad/agrad.hpp"
#include "stan/agrad/matrix.hpp"
#include "stan/agrad/agrad_special_functions.hpp"
#include "stan/prob/distributions.hpp"
#include "stan/maths/special_functions.hpp"
#include "stan/mcmc/hmc.hpp"
#include "stan/model/prob_grad_ad.hpp"
#include <cmath>
#include <ctime>
#include <vector>
#include <stdio.h>

typedef stan::agrad::var rv;
typedef Eigen::Matrix<int,Eigen::Dynamic,1> vec_int;

using namespace stan::agrad;
using namespace stan::maths;
using namespace stan::prob;

class logistic_regression  : public stan::mcmc::prob_grad_ad {

public:
  
  logistic_regression(mat_double x,
                      vec_int y,
                      double scale) :
    stan::mcmc::prob_grad_ad(y.rows()),
    x_(x),
    y_(y),
    scale_(scale) {
  }

  rv log_prob(std::vector<rv>& params_r,
              std::vector<unsigned int>& params_i) {

    // marshal parameters
    vec_var beta(params_r.size());
    for (unsigned int k = 0; k < params_r.size(); ++k)
      beta[k] = params_r[k];
    
    // log prob accumulator
    rv log_prob(0.0);

    // likelihood: log p(x|beta)
    for (unsigned int n = 0; n < x_.rows(); ++n) {
      var basis_n = 0.0;
      vec_double x_n = x_.row(n);
      mat_double x_n_mat = x_n;  
      mat_var beta_mat = beta;
      x_n_mat.transpose() * beta_mat;
      for (unsigned int k = 0; k < params_r.size(); ++k)
        basis_n += beta[k] * x_n[k];
      log_prob += bernoulli_log(y_[n],inv_logit(basis_n));
    }
    
    // prior: log p(beta)
    static double location = 0.0;
    for (unsigned int k = 0; k < params_r.size(); ++k)
      log_prob += cauchy_log(beta[k], location, scale_);

    return log_prob;
  }
  
private: 
  const mat_double x_;
  const vec_int y_;
  const double scale_;

};

int main() {
  printf("hello there");
}
