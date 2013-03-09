#ifndef __STAN__MCMC__UTIL_HPP__
#define __STAN__MCMC__UTIL_HPP__

#include <cstddef>
#include <stdexcept>
#include <fstream>

#include <boost/random/uniform_01.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/exception/diagnostic_information.hpp> 
#include <boost/exception_ptr.hpp> 

#include <stan/math/util.hpp>
#include <stan/model/prob_grad.hpp>

namespace stan {

  namespace mcmc {

    void write_error_msgs(std::ostream* error_msgs,
                          const std::domain_error& e) {
      if (!error_msgs) return;
      *error_msgs << std::endl
                  << "Informational Message: The parameter state is about to be Metropolis"
                  << " rejected due to the following underlying, non-fatal (really)"
                  << " issue (and please ignore that what comes next might say 'error'): "
                  << e.what()
                  << std::endl
                  << "If the problem persists across multiple draws, you might have"
                  << " a problem with an initial state or a gradient somewhere."
                  << std::endl
                  << " If the problem does not persist, the resulting samples will still"
                  << " be drawn from the posterior."
                  << std::endl;
        }


    /**
     * Computes the log probability for a single leapfrog step 
     * in Hamiltonian Monte Carlo.
     *
     * If a domain error occurs when calling the model's grad_log_prob(), 
     * this function returns -inf. Domain errors
     * can occur when distribution functions are called with parameters
     * out of support.
     *
     * @param[in] model Probability model with gradients.
     * @param[in] z Integer parameters.
     * @param[in] x Real parameters
     * @param[in,out] m Momentum.
     * @param[in,out] g Gradient at x, z.
     * @param[in] epsilon Step size used in Hamiltonian dynamics.
     * @param[in,out] error_msgs Output stream for error messages.
     * @param[in,out] output_msgs Output stream for output messages.
     * 
     * @return the log probability of x and m.
     */
    double leapfrog(stan::model::prob_grad& model, 
                    std::vector<int> z,
                    std::vector<double>& x, std::vector<double>& m,
                    std::vector<double>& g, double epsilon,
                    std::ostream* error_msgs = 0,
                    std::ostream* output_msgs = 0) {
      stan::math::scaled_add(m, g, 0.5 * epsilon);
      stan::math::scaled_add(x, m, epsilon);
      double logp;
      try {
        logp = model.grad_log_prob(x, z, g, output_msgs);
      } catch (std::domain_error e) {
        write_error_msgs(error_msgs,e);
        logp = -std::numeric_limits<double>::infinity();
      }
      stan::math::scaled_add(m, g, 0.5 * epsilon);
      return logp;
    }

    // Returns the new log probability of x and m
    // Catches domain errors and sets logp as -inf.
    // Uses a different step size for each variable in x and m.
    double rescaled_leapfrog(stan::model::prob_grad& model, 
                             std::vector<int> z, 
                             const std::vector<double>& step_sizes,
                             std::vector<double>& x, std::vector<double>& m,
                             std::vector<double>& g, double epsilon,
                             std::ostream* error_msgs = 0,
                             std::ostream* output_msgs = 0) {
      for (size_t i = 0; i < m.size(); i++)
        m[i] += 0.5 * epsilon * step_sizes[i] * g[i];
      for (size_t i = 0; i < x.size(); i++)
        x[i] += epsilon * step_sizes[i] * m[i];
      double logp;
      try {
        logp = model.grad_log_prob(x, z, g, output_msgs);
      } catch (std::domain_error e) {
        write_error_msgs(error_msgs,e);
        logp = -std::numeric_limits<double>::infinity();
      }
      for (size_t i = 0; i < m.size(); i++)
        m[i] += 0.5 * epsilon * step_sizes[i] * g[i];
      return logp;
    }
      
      // Uses a nondiagonal mass matrix to conduct leapfrog
      double nondiag_leapfrog(stan::model::prob_grad& model,
                              std::vector<int> z,
                              const Eigen::MatrixXd& _cov_L,
                              std::vector<double>& x, std::vector<double>& m,
                              std::vector<double>& g, double epsilon,
                              std::ostream* error_msgs = 0,
                              std::ostream* output_msgs = 0) {
          Eigen::Map<Eigen::VectorXd> x_mat(&x[0],x.size());
          Eigen::Map<Eigen::VectorXd> m_mat(&m[0],m.size());
          Eigen::Map<Eigen::VectorXd> g_mat(&g[0],g.size());
          m_mat += (0.5 * epsilon) * (_cov_L.transpose().triangularView<Eigen::Upper>() * g_mat);
          x_mat += epsilon * (_cov_L.triangularView<Eigen::Lower>() * m_mat);
          double logp;
          try {
              logp = model.grad_log_prob(x, z, g, output_msgs);
          } catch (std::domain_error e) {
              write_error_msgs(error_msgs,e);
              logp = -std::numeric_limits<double>::infinity();
          }
          m_mat += (0.5 * epsilon) * (_cov_L.transpose().triangularView<Eigen::Upper>() * g_mat);
          return logp;
      }
      
      
      void read_cov(std::string& cov_file,
                    Eigen::MatrixXd& cov_L)
      {
          std::fstream cov_stream(cov_file.c_str());
          for(int i = 0; i < cov_L.rows(); i++){
              for(int j=0; j< cov_L.cols(); j++){
                  cov_stream >> cov_L(i,j);
              }
          }
          //cov_stream.read((char *)_cov_L.data(), sizeof(Eigen::MatrixXd::Scalar)*_cov_L.size());
          // stan::io::dump data_var_context(mass_stream);
          cov_stream.close();
          cov_L = cov_L.selfadjointView<Eigen::Upper>().llt().matrixL();
          
      }



    // this is for eventual gibbs sampler for discrete
    int sample_unnorm_log(std::vector<double> probs, 
                          boost::uniform_01<boost::mt19937&>& rand_uniform_01) {
      // linearize and scale, but don't norm
      double mx = stan::math::max(probs);
      for (size_t k = 0; k < probs.size(); ++k)
        probs[k] = exp(probs[k] - mx);

      // norm by scaling uniform sample
      double sum_probs = stan::math::sum(probs);
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
