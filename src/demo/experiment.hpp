#ifndef __EXPERIMENT__HPP__
#define __EXPERIMENT__HPP__

#include <time.h>

#include <stan/mcmc/adaptive_hmc.hpp>
#include <stan/mcmc/adaptive_cdhmc.hpp>
#include <stan/mcmc/nuts.hpp>
#include <stan/mcmc/prob_grad.hpp>

struct ExperimentParams {
public:
  int _random_seed, _L, _adapttime, _num_samples, _num_args;
  double _epsilon, _delta, _epsilonL;
  char _samplertype;
  stan::mcmc::adaptive_sampler* _sampler;

  ExperimentParams(int argc, char** argv, int num_args, 
                   stan::mcmc::prob_grad& model) {
    _num_args = num_args;
    argv += num_args + 1;
    _samplertype = argv[0][0];
    _delta = atof(argv[1]);
    _random_seed = atoi(argv[2]);
    _adapttime = atoi(argv[3]);
    _num_samples = atoi(argv[4]);
    fprintf(stderr, "%c, %f, %d, %d, %d\n", _samplertype, _delta, _random_seed,
            _adapttime, _num_samples);
    assert(_adapttime < _num_samples);

    switch (_samplertype) {
    case 'h':
      _L = atoi(&argv[0][1]);
      _sampler = new stan::mcmc::adaptive_hmc(model, _L, _delta, -1, 
                                              _random_seed);
      break;
    case 'c':
      _epsilonL = atof(&argv[0][1]);
      _sampler = new stan::mcmc::adaptive_cdhmc(model, _epsilonL, _delta, -1,
                                                _random_seed);
      break;
    case 's':
      _sampler = new stan::mcmc::nuts(model, _delta, -1, _random_seed);
      break;
    default:
      fprintf(stderr, "ERROR: %c is not a valid sampler type.\n",_samplertype);
      exit(1);
    }
  }
};

void print_sample(std::vector<double> sample_vec) {
  fprintf(stdout, "%f", sample_vec[0]);
  for (unsigned int i = 1; i < sample_vec.size(); i++)
    fprintf(stdout, " %f", sample_vec[i]);
  fprintf(stdout, "\n");
}

void print_tunables(std::vector<double> tunable_vec) {
  for (unsigned int i = 0; i < tunable_vec.size(); i++)
    fprintf(stderr, " tunable %d: %f", i, tunable_vec[i]);
  fprintf(stderr, "\n");
}

void run_experiment(stan::mcmc::adaptive_sampler* sampler,
                    int num_samples, int adapttime) {
  sampler->adapt_on();

  for (int m = 0; m < num_samples; m++) {
    if (m == adapttime)
      sampler->adapt_off();
    
    int last_nfevals = sampler->nfevals();
    stan::mcmc::sample sample = sampler->next();
    int nfevals = sampler->nfevals();
    std::vector<double> params_r;
    sample.params_r(params_r);
    fprintf(stderr, "sample %d:\tmean_stat = %f\tnfevals=%d\t%d\tlogp = %f\t",
            m, sampler->mean_stat(), nfevals - last_nfevals, nfevals,
            sample.log_prob());

    std::vector<double> tunable_vec;
    sampler->get_parameters(tunable_vec);
    print_tunables(tunable_vec);

    print_sample(params_r);
  }

  fprintf(stderr, "nfevals = %d\n", sampler->nfevals());
}

#endif
