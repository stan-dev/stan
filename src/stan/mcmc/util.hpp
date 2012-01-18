#ifndef __STAN__MCMC__UTIL_HPP__
#define __STAN__MCMC__UTIL_HPP__


#include <boost/random/uniform_01.hpp>
#include <boost/random/mersenne_twister.hpp>

#include <stan/maths/util.hpp>
#include <stan/mcmc/prob_grad.hpp>

namespace stan {

  namespace mcmc {

    // Returns the new log probability of x and m
    double leapfrog(prob_grad& model, 
                    std::vector<int> z,
                    std::vector<double>& x, std::vector<double>& m,
                    std::vector<double>& g, double epsilon) {
      stan::maths::scaled_add(m, g, 0.5 * epsilon);
      stan::maths::scaled_add(x, m, epsilon);
      double logp = model.grad_log_prob(x, z, g);
      stan::maths::scaled_add(m, g, 0.5 * epsilon);
      return logp;
    }

    int sample_unnorm_log(std::vector<double> probs, 
                          boost::uniform_01<boost::mt19937&>& rand_uniform_01) {
      // linearize and scale, but don't norm
      double mx = stan::maths::max_vec(probs);
      for (unsigned int k = 0; k < probs.size(); ++k)
        probs[k] = exp(probs[k] - mx);

      // norm by scaling uniform sample
      double sum_probs = stan::maths::sum_vec(probs);
      // handles overrun due to arithmetic imprecision
      double sample_0_sum = std::max(rand_uniform_01() * sum_probs, sum_probs);  
      int k = 0;
      double cum_unnorm_prob = probs[0];
      while (cum_unnorm_prob < sample_0_sum)
        cum_unnorm_prob += probs[++k];
      return k;
    }


  }

}

#endif
