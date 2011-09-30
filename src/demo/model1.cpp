#include <cmath>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <boost/math/special_functions.hpp>
#include "stan/agrad/agrad.hpp"
#include "stan/agrad/agrad_special_functions.hpp"
#include "stan/mcmc/prob_grad_ad.hpp"
#include "stan/mcmc/hmc.hpp"
#include "stan/mcmc/sampler.hpp"

const double PI = std::atan(1.0)*4;

class model_1 : public stan::mcmc::prob_grad_ad {
public:

  model_1(double arate, double xirate, std::vector< std::vector<double> >& x,
          std::vector<double>& y)
    : stan::mcmc::prob_grad_ad::prob_grad_ad(3 + x.size()),
      x_(x),
      y_(y),
      K_(x.size()),
      N_(x[0].size()),
      arate_(arate),
      xirate_(xirate) {
  }

  ~model_1() { }

  template<class T>
  T log_prob_ad_templated(std::vector<T>& params_r,
                          std::vector<unsigned int>& params_i) {
    T logtau = params_r[params_r.size()-3];
    T loga = params_r[params_r.size()-2];
    T logxi = params_r[params_r.size()-1];
    T tau = exp(params_r[params_r.size()-3]);
    T a = exp(params_r[params_r.size()-2]);
    T xi = exp(params_r[params_r.size()-1]);
    T* beta = &params_r[0];
    T result = 0;

    // log p(y | x, beta)
    std::vector<T> betax(N_, 0);
    for (int k = 0; k < K_; k++)
      for (int i = 0; i < N_; i++)
        betax[i] += x_[k][i] * beta[k];
    for (int i = 0; i < N_; i++)
      result += -log(1. + exp(-(2*y_[i]-1) * betax[i]));

    // log p(beta | tau)
    T betasq = 0;
    for (int i = 0; i < K_; i++)
      betasq += beta[i] * beta[i];
    result += -0.5 * tau * betasq - 0.5 * K_ * (log(2*PI) - logtau);

    // log p(tau | a, xi)
    result += a * logtau - a * xi * tau + a * (logxi + loga) - lgamma(a);
//     result += a * logtau - xi * tau + a * logxi - lgamma(a);

    // log p(log(a), log(xi))
    result += -arate_*a - log(arate_);
    result += -xirate_*xi - log(xirate_);

    return result; 
  }

  stan::agrad::var log_prob_ad(std::vector<stan::agrad::var>& params_r,
		    std::vector<unsigned int>& params_i) {
    return log_prob_ad_templated<stan::agrad::var>(params_r, params_i);
  }

  double log_prob(std::vector<double>& params_r,
                  std::vector<unsigned int>& params_i) {
    return log_prob_ad_templated<double>(params_r, params_i);
  }

  double grad_log_prob(std::vector<double>& params_r,
                       std::vector<unsigned int>& params_i,
                       std::vector<double>& gradient) {
    double logtau = params_r[params_r.size()-3];
    double loga = params_r[params_r.size()-2];
    double logxi = params_r[params_r.size()-1];
    double tau = exp(params_r[params_r.size()-3]);
    double a = exp(params_r[params_r.size()-2]);
    double xi = exp(params_r[params_r.size()-1]);
    double* beta = &params_r[0];
    double result = 0;
    gradient.resize(0);
    gradient.resize(params_r.size(), 0);

    // log p(y | x, beta)
    std::vector<double> betax(N_, 0);
    for (int k = 0; k < K_; k++)
      for (int i = 0; i < N_; i++)
        betax[i] += x_[k][i] * beta[k];
    for (int i = 0; i < N_; i++) {
      double logpy = -log(1. + exp(-(2*y_[i]-1) * betax[i]));
      double py = exp(logpy);
      for (int k = 0; k < K_; k++)
        gradient[k] += (2*y_[i]-1) * (1 - py) * x_[k][i];
      result += logpy;
    }

    // log p(beta | tau)
    double betasq = 0;
    for (int i = 0; i < K_; i++) {
      betasq += beta[i] * beta[i];
      gradient[i] -= beta[i] * tau;
    }
    gradient[K_] += -0.5 * tau * betasq + 0.5 * K_;
    result += -0.5 * tau * betasq - 0.5 * K_ * (log(2*PI) - logtau);

    // log p(tau | a, xi)
    gradient[K_] += a - a * xi * tau;
    gradient[K_ + 1] += a * logtau - a * xi * tau + a * (logxi + loga) + a - a * boost::math::digamma(a);
    gradient[K_ + 2] += -a * xi * tau + a;
    result += a * logtau - a * xi * tau + a * (logxi + loga) - lgamma(a);
//     gradient[K_] += a - xi * tau;
//     gradient[K_ + 1] += a * logtau + a * logxi - a * boost::math::digamma(a);
//     gradient[K_ + 2] += -xi * tau + a;
//     result += a * logtau - xi * tau + a * logxi - lgamma(a);

    // log p(log(a), log(xi))
    gradient[K_ + 1] -= arate_ * a;
    gradient[K_ + 2] -= xirate_ * xi;
    result += -arate_*a - log(arate_);
    result += -xirate_*xi - log(xirate_);

    return result;
  }

  int K() { return K_; }
  int N() { return N_; }

protected:
  std::vector< std::vector<double> > x_;
  std::vector<double> y_;
  int K_;
  int N_;
  double arate_;
  double xirate_;
};

void readX(const char* fname, std::vector< std::vector<double> >& x) {
  std::ifstream infile(fname);
  char buffer[512];
  infile.getline(buffer, 512);
  std::istringstream line(buffer);
  while (!line.eof()) {
    double xval;
    line >> xval;
    x.push_back(std::vector<double>());
    x.back().push_back(xval);
  }
  while (!infile.eof()) { 
    int k = 0;
    infile.getline(buffer, 512);
    std::istringstream line(buffer);
    while (!line.eof()) {
      double xval;
      line >> xval;
      x[k].push_back(xval);
      k++;
    }
  }
  infile.close();

  for (int i = 0; i < 10; i++) {
    for (unsigned int k = 0; k < x.size(); k++)
      fprintf(stderr, "%f ", x[k][i]);
    fprintf(stderr, "\n");
  }
}

void interactX(const std::vector< std::vector<double> >& x, 
               std::vector< std::vector<double> >& newx,
               int max_order) {
  newx.resize(0);
  int K = x.size();
  int N = x[0].size();
  if (max_order >= 0)
    // 0th order term
    newx.push_back(std::vector<double>(N, 1));
  if (max_order >= 1) {
    // 1st order term
    for (int k = 0; k < K; k++) {
      newx.push_back(std::vector<double>(x[k]));
      double mean = 0;
      double meansq = 0;
      for (unsigned int j = 0; j < x[k].size(); j++) {
        mean += x[k][j];
        meansq += x[k][j] * x[k][j];
      }
      mean /= N;
      meansq /= N;
      double std = sqrt(meansq - mean * mean);
      for (unsigned int j = 0; j < x[k].size(); j++) {
        newx.back()[j] = (newx.back()[j] - mean) / std;
      }
    }
  }

  // Higher-order terms
  std::vector<int> boundaries(K+1, 0);
  boundaries[0] = 1;
  for (int k = 0; k < K; k++)
    boundaries[k+1] = k+2;
  for (int order = 2; order <= max_order; order++) {
    std::vector<int> newboundaries;
    newboundaries.push_back(newx.size());
    for (int k = 0; k < K; k++) {
      for (int i = boundaries[k]; i < boundaries[K]; i++) {
        newx.push_back(std::vector<double>(N, 0));
        double mean = 0;
        double meansq = 0;
        for (int j = 0; j < N; j++) {
          newx.back()[j] = x[k][j] * newx[i][j];
          mean += newx.back()[j];
          meansq += newx.back()[j] * newx.back()[j];
        }
        mean /= N;
        meansq /= N;
        double std = sqrt(meansq - mean*mean);
        for (int j = 0; j < N; j++)
          newx.back()[j] = (newx.back()[j] - mean)/std;
      }
      newboundaries.push_back(newx.size());
    }
    boundaries = newboundaries;
  }
}

void readY(const char* fname, std::vector<double>& y) {
  std::ifstream infile(fname);
  // char buffer[512];
  while (!infile.eof()) { 
    double xval;
    infile >> xval;
    y.push_back(xval);
  }
  infile.close();

  for (int i = 0; i < 10; i++)
    fprintf(stderr, "%f ", y[i]);
  fprintf(stderr, "\n");
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

// Usage: bin/model1 ../pythonHMC/naes04xcont.dat ../pythonHMC/naes04y.dat interaction_order
int main(int argc, char** argv) {
  // load data
  std::vector< std::vector<double> > x;
  std::vector<double> y;
  readX(argv[1], x);
  readY(argv[2], y);
  for (unsigned int k = 0; k < x.size(); k++)
    x[k].resize(2000);
  y.resize(2000);

  std::vector< std::vector<double> > xinteracted;
  interactX(x, xinteracted, atoi(argv[3]));
  x = xinteracted;
  fprintf(stderr, "x.size() = %d, x[0].size() = %d\n", (int)x.size(), (int)x[0].size());

//   std::vector< std::vector<double> > temp(4, std::vector<double>());
//   for (int i = 0; i < temp.size(); i++) {
//     temp[i].push_back(i+1);
//     temp[i].push_back(i+10);
//   }
//   fprintf(stderr, "\n\n");
//   printMatrix(temp);
//   std::vector< std::vector<double> > temp2;
//   interactX(temp, temp2, 4);
//   fprintf(stderr, "\n\n");
//   printMatrix(temp2);

//   return 1;

  // build model
  double arate = 0.1;
  double xirate = 0.1;
  model_1 model(arate, xirate, x, y);

  // initialize params
  std::vector<double> params_r(model.K()+3, 0);
  std::vector<unsigned int> params_i;

  // Initial log probability
  fprintf(stderr, "log p at t=0 = %f\n", model.log_prob(params_r, params_i));

  // MAP estimation
  // double eta = 0.001 / float(model.N());
//   for (int i = 0; i < 10000; i++) {
//     double logp = model.gradientMethod(params_r, params_i, eta);
//     fprintf(stderr, "%d:   %f\n", i, logp);
//   }
//   double logp = model.FISTA(params_r, params_i, eta, 500);
//   double logp = model.nesterov(params_r, params_i, eta, 2000);
  // double logp = model.conjugateGradient(params_r, params_i, 200);
  fprintf(stderr, "params_r after cg:\n");
  for (unsigned int i = 0; i < params_r.size(); i++)
    fprintf(stderr, "%f ", params_r[i]);
  fprintf(stderr, "\n");

  std::vector<double> gradient;
  double temp1 = model.log_prob(params_r, params_i);
  double temp2 = model.grad_log_prob(params_r, params_i, gradient);
  fprintf(stderr, "log_prob: %f   grad_log_prob: %f\n", temp1, temp2);

//   logp = model.log_prob(params_r, params_i);
//   eta = 0.05;
//   for (int m = 0; m < 1000000; m++) {
//     logp = metropolis(model, params_r, params_i, logp, eta);

//     if (m % 1000 == 0) {
//       fprintf(stderr, "%d:  logp = %f\n", m, logp);
//       for (int i = 0; i < params_r.size(); i++)
//         fprintf(stdout, "%f ", params_r[i]);
//       fprintf(stdout, "\n");
//     }
//   }

//   return 0;

  double epsilon = 1.0;
  int Tau = 19;
  int random_seed = 100001;
  stan::mcmc::hmc sampler(model,epsilon,Tau,random_seed);
  sampler.set_x(params_r);
//   for (int i = 0; i < 100; i++)
//     sampler.tune(5);
  for (int i = 0; i < 30; i++)
    sampler.tune((i+1)*100);
  int num_samples = 10000;
  for (int m = 0; m < num_samples; ++m) {
    stan::mcmc::sample sample = sampler.next();
    std::vector<double> params_r = sample.params_r();
    if (m % 10 == 0) {
      fprintf(stderr, "%d:  logp = %f\n", m, sample.log_prob());
      for (unsigned int i = 0; i < params_r.size(); i++)
        fprintf(stdout, "%f ", params_r[i]);
      fprintf(stdout, "\n");
    }
  }
  

//   // log prob check
//   std::vector<unsigned int> paramsI(0);
//   std::vector<double> paramsR(2);
//   paramsR[0] = -1.0;  
//   paramsR[1] = -1.5;
//   double lp = model.log_prob(paramsR,paramsI);
//   std::cout << "expect dmnorm(c(-1,-1.5),mu,Sigma,log=TRUE) = -31.10868\n";
//   std::cout << "found: " << lp << "\n\n";

//   // gradient check
//   std::vector<double> grad(2);
//   double lpg = model.grad_log_prob(paramsR,paramsI,grad);
//   std::cout << "log prob (via gradient)=" << lpg << "\n";
//   std::cout << "grad[0]=" << grad[0] << "\n";
//   std::cout << "grad[1]=" << grad[1] << "\n";
//   double lpdirect1 = grad1(paramsR[0],paramsR[1],mu1,mu2,sigma1,sigma2,rho);
//   double lpdirect2 = grad1(paramsR[1],paramsR[0],mu2,mu1,sigma2,sigma1,rho);
//   std::cout << "direct-grad[0]=" << lpdirect1 << "\n";
//   std::cout << "direct-grad[1]=" << lpdirect2 << "\n";


}


