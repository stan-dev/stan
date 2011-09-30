#include <assert.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <Sacado.hpp>
#include <prob_grad_ad.h>
#include "prob.h"
#include "hmc.h"

static int NUM_PARAMS_R = 11;
static int NUM_SCHOOLS = 8;

typedef Sacado::Rad::ADvar<double> AD;

class eight_schools_model : public mcmc::prob_grad_ad {

  template <typename T>
  T log_prob_t(std::vector<T>& params_r,
	       std::vector<unsigned int>& params_i) {
    int k = 0;

    T mu = params_r[k++];
    T nu = exp(params_r[k++]); // nu > 0
    T tau = exp(params_r[k++]); // tau > 0
    std::vector<T> theta(NUM_SCHOOLS);
    for (int j = 0; j < NUM_SCHOOLS; ++j)
      theta[j] = params_r[k++];

    T log_prob = 0.0;

    for (int j = 0; j < NUM_SCHOOLS; ++j)
      log_prob += prob::normal_log(_y[j],theta[j],_sigma[j]);
    for (int j = 0; j < NUM_SCHOOLS; ++j)
      log_prob += prob::student_t_log(theta[j],nu,mu,tau);

    log_prob += 2.0 * prob::normal_log(nu,0.0,10.0); // half normal
    
    log_prob += 2.0 * prob::normal_log(1.0/tau,0.0,10.0); // inv half normal

    return log_prob;
  }

public:
  eight_schools_model(std::vector<double> y,
		      std::vector<double> sigma)
    : mcmc::prob_grad_ad::prob_grad_ad(NUM_PARAMS_R),
      _y(y),
      _sigma(sigma) {
    assert(y.size() == NUM_SCHOOLS);
    assert(sigma.size() == NUM_SCHOOLS);
  }

  AD log_prob_ad(std::vector<AD>& params_r,
		 std::vector<unsigned int>& params_i) {
    return log_prob_t(params_r,params_i);
  }

  double log_prob(std::vector<double>& params_r,
		  std::vector<unsigned int>& params_i) {
    return log_prob_t(params_r,params_i);
  }

private:
  std::vector<double> _y;
  std::vector<double> _sigma;

};

int main() {
  double y_arr[] = {     28.0,  8.0, -3.0,  7.0, -1.0,  1.0, 18.0,  12.0 };
  double sigma_arr[] = { 15.0, 10.0, 16.0, 11.0,  9.0, 11.0, 10.0,  18.0 };
  std::vector<double> y(NUM_SCHOOLS);
  std::vector<double> sigma(NUM_SCHOOLS);
  y.assign(y_arr, y_arr + 8);
  sigma.assign(sigma_arr, sigma_arr + NUM_SCHOOLS);

  eight_schools_model model(y,sigma);

  double epsilon = 0.001;
  int Tau = 500;
  int random_seed = 47;
  mcmc::hmc sampler(model,epsilon,Tau,random_seed);
  for (int j = 0; j < 20; ++j)
    sampler.tune(5000);

  int num_samples = 6000;
  for (int m = 0; m < num_samples; ++m) {
    mcmc::sample sample = sampler.next();
    std::vector<double> params_r = sample.params_r();
    double log_prob = sample.log_prob();

    int k = 0;
    std::cout << log_prob;
    std::cout << " mu=" << params_r[k++];
    std::cout << " nu=" << exp(params_r[k++]);
    std::cout << " tau=" << exp(params_r[k++]);
    for (int i = 0; i < NUM_SCHOOLS; ++i) 
      std::cout << " theta[" << i << "]=" << params_r[k++];
    std::cout << "\n";
  }
}

/* RAW DATA
http://www.stat.columbia.edu/~gelman/bugsR/schools.dat
school estimate sd
A  28  15
B   8  10
C  -3  16
D   7  11
E  -1   9
F   1  11
G  18  10
H  12  18
*/
