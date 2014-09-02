#ifndef STAN__COMMON__INITIALIZE_STATE_HPP
#define STAN__COMMON__INITIALIZE_STATE_HPP

#include <string>
#include <iostream>
#include <math.h>

#include <stan/math/matrix/Eigen.hpp>

#include <boost/lexical_cast.hpp>
#include <boost/random/additive_combine.hpp> // L'Ecuyer RNG
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/variate_generator.hpp>

#include <stan/model/util.hpp>
#include <stan/gm/error_codes.hpp>
#include <stan/common/context_factory.hpp>
#include <stan/common/write_error_msg.hpp>

namespace stan {
  namespace common {    
    
    /**
     * Sets initial state to zero
     *
     * @param[out]    cont_params the initialized state. This should be the 
     *                            right size and set to 0.
     * @param[in,out] model       the model. Side effects on model? I'm not
     *                            quite sure
     * @param[in,out] output      output stream for messages
     */
    template <class Model>
    bool initialize_state_zero(Eigen::VectorXd& cont_params,
                               Model& model,
                               std::ostream* output) {
      cont_params.setZero();
      
      double init_log_prob;
      Eigen::VectorXd init_grad = Eigen::VectorXd::Zero(model.num_params_r());
      
      try {
        stan::model::gradient(model, cont_params, init_log_prob, init_grad, output);
      } catch (const std::exception& e) {
        if (output)
          *output << "Rejecting initialization at zero because of gradient failure."
                  << std::endl << e.what() << std::endl;
        return false;
      }
      
      if (!boost::math::isfinite(init_log_prob)) {
        if (output)
          *output << "Rejecting initialization at zero because of vanishing density."
                  << std::endl;
        return false;
      }
      
      for (int i = 0; i < init_grad.size(); ++i) {
        if (!boost::math::isfinite(init_grad[i])) {
          if (output)
            *output << "Rejecting initialization at zero because of divergent gradient."
                    << std::endl;
          return false;
        }
      }
      return true;
    }


    /**
     * Initializes state to random uniform values within range.
     *
     * @param[in]     R           valid range of the initialization; must be
     *                            greater than or equal to 0.
     * @param[out]    cont_params the initialized state. This should be the 
     *                            right size and set to 0.
     * @param[in,out] model       the model. Side effects on model? I'm not
     *                            quite sure
     * @param[in,out] base_rng    the random number generator. State may change.
     * @param[in,out] output      output stream for messages
     */
    template <class Model, class RNG>
    bool initialize_state_random(const double R,
                                 Eigen::VectorXd& cont_params,
                                 Model& model,
                                 RNG& base_rng,
                                 std::ostream* output) {
      int num_init_tries = -1;
      
      boost::random::uniform_real_distribution<double>
        init_range_distribution(-R, R);
          
      boost::variate_generator<RNG&, boost::random::uniform_real_distribution<double> >
        init_rng(base_rng, init_range_distribution);
          
      cont_params.setZero();
          
      // Random initializations until log_prob is finite
      Eigen::VectorXd init_grad = Eigen::VectorXd::Zero(model.num_params_r());
      static int MAX_INIT_TRIES = 100;
          
      for (num_init_tries = 1; num_init_tries <= MAX_INIT_TRIES; ++num_init_tries) {
        for (int i = 0; i < cont_params.size(); ++i) 
          cont_params(i) = init_rng();

        double init_log_prob;
        try {
          stan::model::gradient(model, cont_params, init_log_prob, init_grad, &std::cout);
        } catch (const std::exception& e) {
          write_error_msg(output, e);
          if (output)
            *output << "Rejecting proposed initial value with zero density." << std::endl;
          init_log_prob = -std::numeric_limits<double>::infinity();
        }
        if (!boost::math::isfinite(init_log_prob))
          continue;
        for (int i = 0; i < init_grad.size(); ++i)
          if (!boost::math::isfinite(init_grad(i)))
            continue;
        break;
            
      }
          
      if (num_init_tries > MAX_INIT_TRIES) {
        if (output)
          *output << std::endl << std::endl
                  << "Initialization between (" << -R << ", " << R << ") failed after "
                  << MAX_INIT_TRIES << " attempts. " << std::endl
                  << " Try specifying initial values,"
                  << " reducing ranges of constrained values,"
                  << " or reparameterizing the model."
                  << std::endl;
        return false;
      }
      return true;
    }


    /**
     * Creates the initial state using the source parameter
     *
     * @param[in]     source      a string that the context_factory can 
     *                            interpret and provide a valid var_context
     * @param[out]    cont_params the initialized state. This should be the 
     *                            right size and set to 0.
     * @param[in,out] model       the model. Side effects on model? I'm not
     *                            quite sure
     * @param[in,out] base_rng    the random number generator. State may change.
     * @param[in,out] output      output stream for messages
     * @param[in,out] context_factory  an instantiated factory that implements
     *                            the concept of a context_factory. This has
     *                            one method that takes a string.
     */
    template <class ContextFactory, class Model, class RNG>
    bool initialize_state_source(const std::string source,
                                 Eigen::VectorXd& cont_params,
                                 Model& model,
                                 RNG& base_rng,
                                 std::ostream* output,
                                 ContextFactory& context_factory) {
      try {
        typename ContextFactory::var_context_t context = context_factory(source);
        model.transform_inits(context, cont_params);
      } catch(const std::exception& e) {
        if (output)
          *output << "Initialization from source failed."
                  << std::endl << e.what() << std::endl;
        return false;
      }
      
      double init_log_prob;
      Eigen::VectorXd init_grad = Eigen::VectorXd::Zero(model.num_params_r());
      
      try {
        stan::model::gradient(model, cont_params, init_log_prob, init_grad, &std::cout);
        } catch (const std::exception& e) {
        if (output)
          *output << "Rejecting user-specified initialization because of gradient failure."
                  << std::endl << e.what() << std::endl;
        return false;
      }
      
      if (!boost::math::isfinite(init_log_prob)) {
        if (output)
          *output << "Rejecting user-specified initialization because of vanishing density."
                  << std::endl;
        return false;
      }
      
      for (int i = 0; i < init_grad.size(); ++i) {
        if (!boost::math::isfinite(init_grad[i])) {
          if (output)
            *output << "Rejecting user-specified initialization because of divergent gradient."
                    << std::endl;
          return false;
        }
      }
      return true;
    }
    
    /**
     * Converts string to double. Returns true if it is able to convert
     * the number, false otherwise.
     *
     * @param[in]  s     string input
     * @param[out] val   the double value of the string if it is parsable
     *                   as a double; else NaN
     */
    bool get_double_from_string(const std::string& s, double& val) {
      try {
        val = boost::lexical_cast<double>(s);
      } catch (const boost::bad_lexical_cast& e) {
        val = std::numeric_limits<double>::quiet_NaN();
        return false;
      }
      return true;
    }

    /**
     * Creates the initial state.
     *
     * @param[in]     init        init can either be "0", a number as a string,
     *                            or a filename.
     * @param[out]    cont_params the initialized state. This should be the 
     *                            right size and set to 0.
     * @param[in,out] model       the model. Side effects on model? I'm not
     *                            quite sure
     * @param[in,out] base_rng    the random number generator. State may change.
     * @param[in,out] output      output stream for messages
     * @param[in,out] context_factory  an instantiated factory that implements
     *                            the concept of a context_factory. This has
     *                            one method that takes a string.
     */
    template <class ContextFactory, class Model, class RNG>
    bool initialize_state(const std::string init,
                          Eigen::VectorXd& cont_params,
                          Model& model,
                          RNG& base_rng,
                          std::ostream* output,
                          ContextFactory& context_factory) {
      double R;
      if (get_double_from_string(init, R)) {
        if (R == 0) {
          return initialize_state_zero(cont_params, model, output);
        } else {
          return initialize_state_random(R, cont_params, model,
                                         base_rng, output);
        }
      } else {
        return initialize_state_source(init, cont_params, model,
                                       base_rng, output,
                                       context_factory);
      }
      return false;
    }
  } // common
} // stan

#endif
