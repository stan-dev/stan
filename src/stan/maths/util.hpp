#ifndef __STAN__MATHS__UTIL_HPP__
#define __STAN__MATHS__UTIL_HPP__

#include <vector>

namespace stan {

  namespace util {

    inline double max(double a, double b) { return a > b ? a : b; }

    inline double min(double a, double b) { return a < b ? a : b; }

    // x' * y
    inline double dot(std::vector<double>& x,
                      std::vector<double>& y) {
      double sum = 0.0;
      for (unsigned int i = 0; i < x.size(); ++i)
        sum += x[i] * y[i];
      return sum;
    }

    // x' * x
    inline double dot_self(std::vector<double>& x) {
      double sum = 0.0;
      for (unsigned int i = 0; i < x.size(); ++i)
        sum += x[i] * x[i];
      return sum;
    }

    // x <- x + lambda * y
    inline void scaled_add(std::vector<double>& x, 
                           std::vector<double>& y,
                           double lambda) {
      for (unsigned int i = 0; i < x.size(); ++i)
        x[i] += lambda * y[i];
    }

    inline void sub(std::vector<double>& x, std::vector<double>& y,
                    std::vector<double>& result) {
      result.resize(x.size());
      for (unsigned int i = 0; i < x.size(); ++i)
        result[i] = x[i] - y[i];
    }

    inline double dist(const std::vector<double>& x, const std::vector<double>& y) {
      double result = 0;
      for (unsigned int i = 0; i < x.size(); ++i) {
        double diff = x[i] - y[i];
        result += diff * diff;
      }
      return sqrt(result);
    }

    inline double sum_vec(std::vector<double> x) {
      double sum = x[0];
      for (unsigned int i = 1; i < x.size(); ++i)
        sum += x[i];
      return sum;
    }

    inline double max_vec(std::vector<double> x) {
      double max = x[0];
      for (unsigned int i = 1; i < x.size(); ++i)
        if (x[i] > max)
          max = x[i];
      return max;
    }
      
    int sample_unnorm_log(std::vector<double> probs, boost::uniform_01<boost::mt19937&>& rand_uniform_01) {
      // linearize and scale, but don't norm
      double mx = max_vec(probs);
      for (unsigned int k = 0; k < probs.size(); ++k)
        probs[k] = exp(probs[k] - mx);

      // norm by scaling uniform sample
      double sum_probs = sum_vec(probs);
      // handles overrun due to arithmetic imprecision
      double sample_0_sum = std::max(rand_uniform_01() * sum_probs, sum_probs);  
      int k = 0;
      double cum_unnorm_prob = probs[0];
      while (cum_unnorm_prob < sample_0_sum)
        cum_unnorm_prob += probs[++k];
      return k;
    }

    // Returns the new log probability of x and m
    inline double leapfrog(mcmc::prob_grad& model, 
                           std::vector<int> z,
                           std::vector<double>& x, std::vector<double>& m,
                           std::vector<double>& g, double epsilon) {
      scaled_add(m, g, 0.5 * epsilon);
      scaled_add(x, m, epsilon);
      double logp = model.grad_log_prob(x, z, g);
      scaled_add(m, g, 0.5 * epsilon);
      return logp;
    }
  }
}

#endif
