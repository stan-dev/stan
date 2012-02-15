#include <cmath>
#include <ctime>
#include <vector>
#include <stdio.h>
#include "stan/agrad/agrad.hpp"
#include "stan/mcmc/hmc.hpp"
#include "stan/mcmc/sampler.hpp"
#include "stan/model/prob_grad_ad.hpp"

const double PI = std::atan(1.0)*4;

const unsigned int NUM_PARAMETERS_R = 2;

typedef stan::agrad::var RV;

class bivar_norm_model : public stan::mcmc::prob_grad_ad {
public:
  bivar_norm_model(double mu1, 
                   double mu2,
                   double sigma1, 
                   double sigma2, 
                   double rho)
    : stan::mcmc::prob_grad_ad::prob_grad_ad(NUM_PARAMETERS_R),
      _mu1(mu1), 
      _mu2(mu2),
      _sigma1(sigma1), 
      _sigma2(sigma2), 
      _rho(rho),
      _two_times_one_minus_rho_sq(2.0 * (1.0 - rho * rho)),
      _log_inv_z(- log(2.0 * PI * sqrt(1.0 - rho * rho) * sigma1 * sigma2)) {
  }

  RV log_prob(std::vector<RV>& params_r,
              std::vector<int>& params_i) {
    RV y1 = params_r[0];    
    RV y2 = params_r[1];
    RV z1 = (y1 - _mu1)/_sigma1;
    RV z2 = (y2 - _mu2)/_sigma2;
    RV result = _log_inv_z 
      - ( z1 * z1 
          + z2 * z2 
          - 2.0 * _rho * z1 * z2 )
      / _two_times_one_minus_rho_sq;
    return result; 
  }

private:
  double _mu1;
  double _mu2;
  double _sigma1;
  double _sigma2;
  double _rho;
  double _two_times_one_minus_rho_sq;
  double _log_inv_z;
};

int main() {
  // construct model given constants/data
  double mu1 = 0.0;
  double mu2 = 0.0;
  double sigma1 = 1.0;
  double sigma2 = 1.0;
  double rho = 0.998;
  bivar_norm_model model(mu1,mu2,sigma1,sigma2,rho); 

  // configure HMC (following MacKay's book params)
  double epsilon = 0.055; // step size
  unsigned int Tau = 19;  // number of steps
  // int random_seed = 43; // optional last arg to sampler ctor
  stan::mcmc::hmc sampler(model,epsilon,Tau); 

  
  // sampler.tune(200); // doesn't work so well
  
  std::clock_t t_start = std::clock();
  // printf("tim=%d\n",t_start);

  // sample
  int num_samples = 128; // 100K for speed tests

  for (int m = 0; m < num_samples; ++m) {
    stan::mcmc::sample sample = sampler.next();
    double x = sample.params_r(0);
    double y = sample.params_r(1);
    double log_prob = sample.log_prob();
    printf("sample %4d:  (%+5.3f, %+5.3f)  log prob=%+5.3f\n",
           m, x, y, log_prob);
  }

  std::clock_t t_end = std::clock();
  int t_dif = t_end - t_start;
  printf("elapsed=%8.6f seconds\n",t_dif/(double)CLOCKS_PER_SEC);
  return 0;
}

