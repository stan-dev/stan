#include <cmath>
#include <ctime>
#include <vector>
#include <stdio.h>
#include "stan/agrad/agrad.hpp"
#include "stan/mcmc/prob_grad_ad.hpp"
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

  int random_seed = 100003;
  double epsilon = 1;
  int L = 300;
  int adapttime = 500;
  int num_samples = adapttime + 10000;
  int num_args = 1;
  char samplertype = argv[num_args+1][0];
  stan::mcmc::adaptive_sampler* sampler = NULL;

  double delta = atof(argv[num_args+2]);
  if (argc > num_args+2)
    random_seed = atoi(argv[num_args+3]);
  if (samplertype == 'h') {
    L = atoi(&argv[num_args+1][1]);
    sampler = new stan::mcmc::adaptive_hmc(model, L, delta, -1, random_seed);
  } else if (samplertype == 'c') {
    double epsilonL = atof(&argv[num_args+1][1]);
    sampler = new stan::mcmc::adaptive_cdhmc(model, epsilonL, delta, -1, 
                                             random_seed);
  } else if (samplertype == 'n') {
    sampler = new stan::mcmc::nuts(model, delta, -1, random_seed);
  }
  sampler->adapt_on();
  std::vector<double> epsilon_vec;
  sampler->get_parameters(epsilon_vec);
  fprintf(stderr, "initial epsilon = %f\n", epsilon_vec[0]);

  for (int m = 0; m < num_samples; ++m) {
    if (m >= adapttime)
      sampler->adapt_off();

    stan::mcmc::sample sample = sampler->next();
    std::vector<double> params_r;
    sample.params_r(params_r);
    if (m % 1 == 0) {
      sampler->get_parameters(epsilon_vec);
      fprintf(stderr, "%d:\tepsilon = %f\tmean_stat = %f\tlogp = %f\n", m, 
              epsilon_vec[0], sampler->mean_stat(), sample.log_prob());
      for (unsigned int i = 0; i < NUM_PARAMETERS_R; i++)
        printf("%s%f", i == 0 ? "" : " ", params_r[i]);
      printf("\n");
    }
  }

  fprintf(stderr, "nfevals = %d\n", sampler->nfevals());

//   int random_seed = 100003;
//   // configure HMC
//   double epsilon = 0.04; // step size
//   int num_samples = 1000;
//   int adapttime = num_samples / 2;
// //   double Lepsilon = atof(argv[2]);
// //   unsigned int Tau = int(Lepsilon / epsilon);  // number of steps
//   char samplertype = argv[2][0];
//   stan::mcmc::sampler* sampler;
//   if (samplertype == 'h') {
//     unsigned int Tau = atoi(&argv[2][1]);  // number of steps
//     double delta = atof(argv[3]);
//     if (argc > 4)
//       random_seed = atoi(argv[4]);
//     sampler = new stan::mcmc::adaptivehmc(model, epsilon, Tau, delta, 
//                                           adapttime, random_seed);
// //     sampler = new stan::mcmc::hmc(model, epsilon, Tau, random_seed);
//   } else if (samplertype == 'n') {
//     double delta = atof(&argv[2][1]);
//     if (argc > 3)
//       random_seed = atoi(argv[3]);
//     sampler = new stan::mcmc::nuts(model, epsilon, delta, random_seed);
//   } else {
//     fprintf(stderr, "unrecognized sampler type %c\n", argv[2][0]);
//     return 1;
//   }
  
  return 0;
}

