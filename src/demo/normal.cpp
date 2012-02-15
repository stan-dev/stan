#include <cmath>
#include <ctime>
#include <vector>
#include <stdio.h>
#include "demo/experiment.hpp"
#include "stan/agrad/agrad.hpp"
#include "stan/model/prob_grad_ad.hpp"
#include "stan/mcmc/hmc.hpp"
#include "stan/mcmc/adaptive_hmc.hpp"
#include "stan/mcmc/adaptive_cdhmc.hpp"
#include "stan/mcmc/nuts.hpp"

const double PI = std::atan(1.0)*4;

const unsigned int D = 250;
const unsigned int NUM_PARAMETERS_R = D;

typedef stan::agrad::var RV;

void read_data(const char* fname, std::vector<double>& x) {
  x.resize(D*D);
  double* xptr = &x[0];
  FILE* fptr = fopen(fname, "r");
  for (unsigned int i = 0; i < D*D; i++) {
    fscanf(fptr, "%lf", xptr);
    xptr++;
  }
}

class norm_model : public stan::mcmc::prob_grad_ad {
public:

  // construct model, storing data and precomputing constant terms
  norm_model(const char* Afile)
    : stan::mcmc::prob_grad_ad::prob_grad_ad(NUM_PARAMETERS_R)
  {
    read_data(Afile, A_);
  }

  // compute log prob using typedef for automatic gradient calc
  RV log_prob(std::vector<RV>& params_r,
              std::vector<int>& params_i) {
    RV result = 0;
    for (unsigned int i = 0; i < D; i++) {
      double* Ai = &A_[i*D];
      RV temp = 0;
      for (unsigned int j = 0; j < D; j++)
        temp += Ai[j] * params_r[j];
      result += params_r[i] * temp;
    }
    result *= -0.5;

    return result; 
  }

  double grad_log_prob(std::vector<double>& params_r,
                       std::vector<int>& params_i,
                       std::vector<double>& gradient) {
    gradient.assign(num_params_r(), 0);
    double result = 0;
    for (unsigned int i = 0; i < D; i++) {
      double* Ai = &A_[i*D];
      double temp = 0;
      for (unsigned int j = 0; j < D; j++) {
        temp += Ai[j] * params_r[j];
        gradient[i] -= params_r[j] * Ai[j];
      }
      result += params_r[i] * temp;
    }
    result *= -0.5;

    return result; 
  }

private:
  std::vector<double> A_;
};

int main(int argc, char** argv) {
  norm_model model(argv[1]);

  ExperimentParams params(argc, argv, 1, model);

  run_experiment(params._sampler, params._num_samples, params._adapttime);
  
  return 0;
}

