#include <cmath>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <boost/math/special_functions.hpp>
#include "stan/agrad/agrad.hpp"
#include "stan/agrad/agrad_special_functions.hpp"
#include "stan/model/prob_grad_ad.hpp"
#include "stan/mcmc/adaptivehmc.hpp"
#include "stan/mcmc/nutsshort2.hpp"
#include "stan/mcmc/sampler.hpp"

const double PI = std::atan(1.0)*4;

class lda : public stan::mcmc::prob_grad_ad {
protected:
  int D_, W_, K_;
  std::vector< std::vector<int> > words_;
  std::vector< std::vector<int> > counts_;
  double alpha_, eta_;

public:
  lda(const std::vector< std::vector<int> > words,
      const std::vector< std::vector<int> > counts,
      double alpha, double eta, int K, int W)
    : stan::mcmc::prob_grad_ad::prob_grad_ad(words.size()*K+K*W),
      D_(words.size()), W_(W), K_(K), words_(words), counts_(counts), 
      alpha_(alpha), eta_(eta) {
  }

//   template<class T>
//   T log_prob_ad_vectorized(std::vector<T>& params_r, 
//                            std::vector<unsigned int>& params_i) {
//     T* theta = &params_r[0];
//     T* beta = &params_r[D_*K_];

//     T result = 0;

//     std::vector<T> betanorm(K_*W_);
//     std::vector<T> normalizers(K_, 0);
//     for (int w = 0; w < W_; ++w) {
//       T* betaw = &beta[w*K_];
//       T* betanormw = &betanorm[w*K_];
//       for (int k = 0; k < K_; ++k) {
//         betanormw[k] = exp(betaw[k]);
//         normalizers[k] += betanormw[k];
//       }
//     }
//     result += eta_ * stan::agrad::sum(beta, K_*W_);
//     result -= stan::agrad::sum(betanorm);
//     for (int k = 0; k < K_; ++k) {
//       normalizers[k] = 1.0 / normalizers[k];
//     }
//     for (int w = 0; w < W_; ++w) {
//       T* betanormw = &betanorm[w*K_];
//       for (int k = 0; k < K_; ++k)
//         betanormw[k] *= normalizers[k];
//     }
//     result -= K_*W_ * lgamma(eta_);

//     std::vector<T> thetanorm(D_*K_);
//     for (int d = 0; d < D_; ++d) {
//       T* thetad = &theta[d*K_];
//       T* thetanormd = &thetanorm[d*K_];
//       for (int k = 0; k < K_; ++k) {
//         thetanormd[k] = exp(thetad[k]);
//       }
//       T normalizer = stan::agrad::sum(thetanormd, K_);
//       result -= normalizer;
//       normalizer = 1.0 / normalizer;
//       for (int k = 0; k < K_; ++k)
//         thetanormd[k] *= normalizer;
//     }
//     result += alpha_ * stan::agrad::sum(theta, D_*K_);
//     result -= D_*K_ * lgamma(alpha_);

//     std::vector<T> pw;
//     std::vector<T> pw2;
//     for (int d = 0; d < D_; ++d) {
//       T* thetad = &thetanorm[d*K_];
//       for (int i = 0; i < words_[d].size(); ++i) {
//         int w = words_[d][i];
//         int n = counts_[d][i];
//         T* betaw = &betanorm[w*K_];
//         pw.push_back(stan::agrad::dot(thetad, betaw, K_));
//         pw.back() = double(n) * log(pw.back());
// //         pw2.push_back(stan::agrad::dot(thetad, betaw, K_));
//       }
//     }
//     result += stan::agrad::sum(pw);

//     return result;
//   }

  template<class T>
  T log_prob_ad_templated(std::vector<T>& params_r, 
                          std::vector<unsigned int>& params_i) {
    T* theta = &params_r[0];
    T* beta = &params_r[D_*K_];

    T result = 0;

    std::vector<T> betanorm(K_*W_);
    std::vector<T> normalizers(K_, 0);
    T logbetasum = 0;
    T betasum = 0;
    for (int w = 0; w < W_; ++w) {
      T* betaw = &beta[w*K_];
      T* betanormw = &betanorm[w*K_];
      for (int k = 0; k < K_; ++k) {
        betanormw[k] = exp(betaw[k]);
        logbetasum += betaw[k];
        result -= betanormw[k];
        normalizers[k] += betanormw[k];
      }
    }
    result += eta_ * logbetasum;
    for (int k = 0; k < K_; ++k) {
      normalizers[k] = 1.0 / normalizers[k];
    }
    for (int w = 0; w < W_; ++w) {
      T* betanormw = &betanorm[w*K_];
      for (int k = 0; k < K_; ++k)
        betanormw[k] *= normalizers[k];
    }
    result -= K_*W_ * lgamma(eta_);

    std::vector<T> thetanorm(D_*K_);
    T logthetasum = 0;
    for (int d = 0; d < D_; ++d) {
      T normalizer = 0;
      T* thetad = &theta[d*K_];
      T* thetanormd = &thetanorm[d*K_];
      for (int k = 0; k < K_; ++k) {
        thetanormd[k] = exp(thetad[k]);
        logthetasum += thetad[k];
        normalizer += thetanormd[k];
      }
      result -= normalizer;
      normalizer = 1.0 / normalizer;
      for (int k = 0; k < K_; ++k)
        thetanormd[k] *= normalizer;
    }
    result += alpha_ * logthetasum;
    result -= D_*K_ * lgamma(alpha_);

    for (int d = 0; d < D_; ++d) {
      T* thetad = &thetanorm[d*K_];
      for (int i = 0; i < words_[d].size(); ++i) {
        int w = words_[d][i];
        int n = counts_[d][i];
        T* betaw = &betanorm[w*K_];
        T pw = 0;
        for (int k = 0; k < K_; k++)
          pw = fma(thetad[k], betaw[k], pw);
//           pw += thetad[k] * betaw[k];
        result += double(n) * log(pw);
      }
    }

    return result;
  }

  double grad_log_prob(std::vector<double>& params_r, 
                       std::vector<unsigned int>& params_i,
                       std::vector<double>& gradient) {
    double* logtheta = &params_r[0];
    double* logbeta = &params_r[D_*K_];
    gradient.assign(params_r.size(), 0);
    double* logthetag = &gradient[0];
    double* logbetag = &gradient[D_*K_];

    double logp = 0;

    std::vector<double> beta(K_*W_);
    std::vector<double> betanormalizers(K_, 0);
    std::vector<double> betanorm(K_*W_);
    for (int w = 0; w < W_; ++w) {
      double* logbetaw = &logbeta[w*K_];
      double* betaw = &beta[w*K_];
      double* logbetagw = &logbetag[w*K_];
      for (int k = 0; k < K_; ++k) {
        betaw[k] = exp(logbetaw[k]);
        betanormalizers[k] += betaw[k];
        logp += eta_ * logbetaw[k];
        logp -= betaw[k];
        logbetagw[k] += eta_ / betaw[k] - 1;
      }
    }
    for (int k = 0; k < K_; ++k) {
      betanormalizers[k] = 1.0 / betanormalizers[k];
    }
    for (int w = 0; w < W_; ++w) {
      double* betanormw = &betanorm[w*K_];
      double* betaw = &beta[w*K_];
      for (int k = 0; k < K_; ++k)
        betanormw[k] = betaw[k] * betanormalizers[k];
    }
    logp -= K_*W_ * lgamma(eta_);

    std::vector<double> theta(D_*K_);
    std::vector<double> thetanormalizers(D_, 0);
    std::vector<double> thetanorm(D_*K_);
    for (int d = 0; d < D_; ++d) {
      double* logthetad = &logtheta[d*K_];
      double* thetad = &theta[d*K_];
      double* logthetagd = &logthetag[d*K_];
      for (int k = 0; k < K_; ++k) {
        thetad[k] = exp(logthetad[k]);
        thetanormalizers[d] += thetad[k];
        logp += alpha_ * logthetad[k];
        logp -= thetad[k];
        logthetagd[k] += alpha_ / thetad[k] - 1;
      }
    }
    for (int d = 0; d < D_; ++d) {
      thetanormalizers[d] = 1.0 / thetanormalizers[d];
    }
    for (int d = 0; d < D_; ++d) {
      double* thetanormd = &thetanorm[d*K_];
      double* thetad = &theta[d*K_];
      for (int k = 0; k < K_; ++k)
        thetanormd[k] = thetad[k] * thetanormalizers[d];
    }
    logp -= D_*K_ * lgamma(alpha_);

    double likelihood = 0;
    std::vector<double> betagnorm(K_, 0);
    for (int d = 0; d < D_; ++d) {
      double thetagnorm = 0;
      double* thetanormd = &thetanorm[d*K_];
      double* logthetagd = &logthetag[d*K_];
      double* thetad = &theta[d*K_];
      for (int i = 0; i < words_[d].size(); ++i) {
        int w = words_[d][i];
        double n = counts_[d][i];
        double* betanormw = &betanorm[w*K_];
        double* logbetagw = &logbetag[w*K_];
        double pw = 0;
        for (int k = 0; k < K_; ++k)
          pw += thetanormd[k] * betanormw[k];
        double pwinv = 1.0 / pw;
        for (int k = 0; k < K_; ++k) {
//           logthetagd[k] += n * betanormw[k] * pwinv;
//           logthetagd[k] += n * betanormw[k] * pwinv * thetanormalizers[d]
//             * (1 - thetanormd[k]);
          logthetagd[k] += n * thetanormalizers[d] * (betanormw[k] * pwinv - 1);
          logbetagw[k] += n * thetanormd[k] * pwinv * betanormalizers[k];
          betagnorm[k] -= n * betanormw[k] * thetanormd[k] * pwinv * betanormalizers[k];
          thetagnorm -= n * betanormw[k] * thetanormd[k] * pwinv;
        }
        logp += n * log(pw);
//         likelihood += n * log(pw);
      }
      for (int k = 0; k < K_; ++k) {
//         logthetagd[k] += thetanormalizers[d] * thetagnorm;
//         logthetagd[k] *= thetad[k] * thetanormalizers[d];
        logthetagd[k] *= thetad[k];
      }
    }
    for (int w = 0; w < W_; ++w) {
      double* logbetagw = &logbetag[w*K_];
      double* betaw = &beta[w*K_];
      for (int k = 0; k < K_; ++k) {
        logbetagw[k] += betagnorm[k];
        logbetagw[k] *= betaw[k];
//         logbetagw[k] *= betaw[k] * betanormalizers[k];
      }
    }
//     fprintf(stderr, "likelihood = %f\n", likelihood);

    return logp;
  }

  stan::agrad::var log_prob(std::vector<stan::agrad::var>& params_r,
                            std::vector<unsigned int>& params_i) {
    return log_prob_ad_templated<stan::agrad::var>(params_r, params_i);
//     if (1)
//       return log_prob_ad_templated<stan::agrad::var>(params_r, params_i);
//     else
//       return log_prob_ad_vectorized<stan::agrad::var>(params_r, params_i);
  }

  double log_prob(std::vector<double>& params_r,
                  std::vector<unsigned int>& params_i) {
    return log_prob_ad_templated<double>(params_r, params_i);
//     if (1)
//       return log_prob_ad_templated<double>(params_r, params_i);
//     else
//       return log_prob_ad_vectorized<double>(params_r, params_i);
  }
};

void read_data(const char* fname, std::vector< std::vector<int> >& words,
               std::vector< std::vector<int> >& counts) {
  words.resize(0);
  counts.resize(0);
  FILE* fptr = fopen(fname, "r");
  int length;
  while (fscanf(fptr, "%10d", &length) != EOF) {
    words.push_back(std::vector<int>(length));
    counts.push_back(std::vector<int>(length));
    for (int i = 0; i < length; ++i) {
      fscanf(fptr, "%10d:%10d", &words.back()[i], &counts.back()[i]);
    }
  }
}

void printMatrix(std::vector< std::vector<double> > x) {
  for (unsigned int i = 0; i < x.size(); i++) {
    for (unsigned int j = 0; j < x[0].size(); j++) {
      fprintf(stderr, "%f ", x[i][j]);
    }
    fprintf(stderr, "\n");
  }
}

double metropolis(stan::mcmc::prob_grad_ad& model, std::vector<double>& params_r,
                  std::vector<unsigned int> params_i, double lastlogp,
                  double eta) {
  static boost::mt19937 rand_int(100001);
  static boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > rand_unit_norm(rand_int, boost::normal_distribution<>());
  static boost::uniform_01<boost::mt19937&> rand_uniform_01(rand_int);

  std::vector<double> newx = params_r;
  for (unsigned int i = 0; i < params_r.size(); i++)
    newx[i] += eta * rand_unit_norm();
  newx[12] += 10*eta * rand_unit_norm();
  newx[13] += 10*eta * rand_unit_norm();
  double newlogp = model.log_prob(newx, params_i);

  if (log(rand_uniform_01()) < newlogp - lastlogp) {
    params_r.assign(newx.begin(), newx.end());
    return newlogp;
  } else {
    return lastlogp;
  }
}

// Usage: bin/lda /path/to/mult.dat K
int main(int argc, char** argv) {
  // load data
  std::vector< std::vector<int> > words;
  std::vector< std::vector<int> > counts;
  int K = atoi(argv[2]);
  read_data(argv[1], words, counts);

  int D = words.size();
  int W = 0;
  int totalN = 0;
  for (int d = 0; d < words.size(); d++) {
    for (int i = 0; i < words[d].size(); i++) {
      if (words[d][i] > W)
        W = words[d][i];
      totalN += counts[d][i];
    }
  }
  ++W;
//   for (int d = 0; d < words.size(); d++) {
//     printf("%d: ", words[d].size());
//     for (int i = 0; i < words[d].size(); i++)
//       printf("%d:%d ", words[d][i], counts[d][i]);
//     printf("\n");
//   }

  double alpha = 1.0 / K;
  double eta = 1.0 / K;
  std::vector<double> params_r(D*K + K*W);
  std::vector<unsigned int> params_i;

  boost::mt19937 rand_int(100001);
  boost::uniform_01<boost::mt19937&> rand01(rand_int);
  for (int i = 0; i < params_r.size(); i++)
    params_r[i] = rand01() - 0.5;

  lda model(words, counts, alpha, eta, K, W);
  std::vector<double> g(params_r.size());
  double logp = model.grad_log_prob(params_r, params_i, g);
  fprintf(stderr, "initial logp = %f\n", logp);
  logp = model.nesterov(params_r, params_i, 0.00075, 200);
//   model.testGradients(params_r, params_i, 1e-6);
//   double logp = conjugate_gradient(params_r, 
//     double conjugate_gradient(std::vector<double>& x,
//                            std::binary_function<vector<double>&, 
//                                                 vector<double>&, 
//                                                 double> grad_function,
//                            int niterations) {  fprintf(stderr, "params_r after cg:\n");

  int random_seed = 100001;
  stan::mcmc::nutsshort2 sampler(model, 0.04, 200, 0.25, random_seed);
//   stan::mcmc::mshmc2 sampler(model, 0.1, random_seed);
  sampler.set_params(params_r, params_i);
  int num_samples = 10000;
  for (int m = 0; m < num_samples; ++m) {
    stan::mcmc::sample sample = sampler.next();
    std::vector<double> params_r;
    sample.params_r(params_r);
    if (m % 1 == 0) {
      fprintf(stderr, "%d:  mean logp = %f\n", m, sample.log_prob()/totalN);
      for (unsigned int i = 0; i < params_r.size(); i++)
        fprintf(stdout, "%f ", params_r[i]);
      fprintf(stdout, "\n");
    }
  }
}


