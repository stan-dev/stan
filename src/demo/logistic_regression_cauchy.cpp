#include "stan/model/prob_grad_ad.hpp"
#include "stan/mcmc/hmc.hpp"

typedef stan::agrad::var RV;
typedef Eigen::Matrix<double,Dynamic,Dynamic> mat_double;
typedef Eigen::Matrix<double,Dynamic,1> vec_double;

class logistic_regression_cauchy : public stan::mcmc::prob_grad_ad {
private: 
  mat_double x;
  vec_double y;

public:

  logistic_regression_cauchy(mat_double x,
                             vec_double y) :
  x_(x), 
  y_(y) {
    assert(x.rows()==y.rows());
  }

  RV log_prob(std::vector<RV>& params_r,
              std::vector<unsigned int>& params_i) {
    RV log_prob(0.0);
    vec_double beta(params_r.size());
    for (int k = 0; k < params_r.size(); ++k)
      beta[i] = params_r[k];
  }

  

    
}
