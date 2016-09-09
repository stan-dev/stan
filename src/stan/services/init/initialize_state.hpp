#ifndef STAN_SERVICES_INIT_INITIALIZE_STATE_HPP
#define STAN_SERVICES_INIT_INITIALIZE_STATE_HPP

#include <boost/lexical_cast.hpp>
#include <boost/random/additive_combine.hpp>  // L'Ecuyer RNG
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <stan/model/gradient.hpp>
#include <stan/services/error_codes.hpp>
#include <stan/io/array_var_context.hpp>
#include <stan/io/chained_var_context.hpp>
#include <stan/interface_callbacks/var_context_factory/var_context_factory.hpp>
#include <stan/interface_callbacks/writer/base_writer.hpp>
#include <stan/services/io/write_error_msg.hpp>
#include <stan/math/prim/mat.hpp>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace stan {
  namespace services {
    namespace init {
      namespace {

        /**
         * Remove indices from a parameter name. Specifically, if a parameter
         * is an array (or similar), the names return from
         * constrained_param_names of a model are like alpha.1, alpha.2. This
         * function removes everything after the first occurrence of a dot.
         *
         * If we can get the parameter names of a model (excluding transformed
         * parameters and generated quantities), this function would not be
         * necessary.
         *
         * FIXME: this function can be removed if the names of
         * parameters excluding transformed parameters and generated
         * quantities can be obtained from a model directly.
         *
         * @param[in,out]  name  parameter name
         */
        void rm_indices_from_name(std::string& name) {
          size_t x = name.find_first_of(".");
          if (std::string::npos == x) return;
          name.erase(x);
        }

        /**
         * Remove indices from parameter names and make names unique.
         *
         * @param[in,out]  names  parameter names
         */
        void rm_indices_from_name(std::vector<std::string>& names) {
          for (size_t i = 0; i < names.size(); i++)
            rm_indices_from_name(names[i]);
          std::vector<std::string>::iterator it;
          it = std::unique(names.begin(), names.end());
          names.resize(std::distance(names.begin(), it));
        }

        /**
         * Check if all the parameters are contained in the
         * user-provided var_context to specify inits
         *
         * FIXME: if the model can provide the names defined
         * in the parameter block, it would be easier.
         *
         * @param[in] model   the model.
         * @param[in] context the context of inits provided by user
         * @return            true if all are contained,
         *                    false if anyone is not
         */

        template <class Model>
        bool are_all_pars_initialized(const Model& model,
                                      const stan::io::var_context& context) {
          std::vector<std::string> names;
          model.constrained_param_names(names, false, false);
          rm_indices_from_name(names);
          for (size_t i = 0; i < names.size(); i++)
            if (!context.contains_r(names[i])) return false;
          return true;
        }

        template <class Model>
        bool validate_unconstrained_initialization(Eigen::VectorXd& cont_params,
                                                   Model& model) {
          for (int n = 0; n < cont_params.size(); n++) {
            if (stan::math::is_inf(cont_params[n])
                || stan::math::is_nan(cont_params[n])) {
              std::vector<std::string> param_names;
              model.unconstrained_param_names(param_names);

              std::stringstream msg;
              msg << param_names[n] << " initialized to invalid value ("
                  << cont_params[n] << ")";

              throw std::invalid_argument(msg.str());
            }
          }
          return true;
        }
      }

      /***
       * Set initial values to what container cont_params has.
       *
       * @param[in]    cont_params the initialized state. This should be the
       *                            right size and set to 0.
       * @param[in,out] model       the model. Side effects on model? I'm not
       *                            quite sure
       * @param[in,out] writer      writer callback for messages
       */
      template <class Model>
      bool initialize_state_values(Eigen::VectorXd& cont_params,
                                   Model& model,
                                   interface_callbacks::writer::base_writer&
                                   writer) {
        try {
          validate_unconstrained_initialization(cont_params, model);
        } catch (const std::exception& e) {
          writer(e.what());
          writer();
          return false;
        }
        double init_log_prob;
        Eigen::VectorXd init_grad = Eigen::VectorXd::Zero(model.num_params_r());
        try {
          stan::model::gradient(model, cont_params, init_log_prob,
                                init_grad, writer);
        } catch (const std::exception& e) {
          io::write_error_msg(writer, e);
          writer("Rejecting initial value:");
          writer("  Error evaluating the log probability "
                 "at the initial value.");
          writer();
          return false;
        }
        if (!boost::math::isfinite(init_log_prob)) {
          writer("Rejecting initial value:");
          writer("  Log probability evaluates to log(0), "
                 "i.e. negative infinity.");
          writer("  Stan can't start sampling from this initial value.");
          writer();
          return false;
        }
        for (int i = 0; i < init_grad.size(); ++i) {
          if (!boost::math::isfinite(init_grad(i))) {
            writer("Rejecting initial value:");
            writer("  Gradient evaluated at the initial value "
                   "is not finite.");
            writer("  Stan can't start sampling from this initial value.");
            writer();
            return false;
          }
        }
        return true;
      }


      /**
       * Sets initial state to zero
       *
       * @param[out]    cont_params the initialized state. This should be the
       *                            right size and set to 0.
       * @param[in,out] model       the model. Side effects on model? I'm not
       *                            quite sure
       * @param[in,out] writer      writer callback for messages
       */
      template <class Model>
      bool initialize_state_zero(Eigen::VectorXd& cont_params,
                                 Model& model,
                             interface_callbacks::writer::base_writer& writer) {
        cont_params.setZero();
        return initialize_state_values(cont_params, model, writer);
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
       * @param[in,out] base_rng    the random number generator.
       *                            State may change.
       * @param[in,out] writer      writer callback for messages
       */
      template <class Model, class RNG>
      bool initialize_state_random(const double R,
                                   Eigen::VectorXd& cont_params,
                                   Model& model,
                                   RNG& base_rng,
                             interface_callbacks::writer::base_writer& writer) {
        int num_init_tries = -1;

        boost::random::uniform_real_distribution<double>
          init_range_distribution(-R, R);

        boost::variate_generator
          <RNG&, boost::random::uniform_real_distribution<double> >
          init_rng(base_rng, init_range_distribution);

        // Random initializations until log_prob is finite
        static int MAX_INIT_TRIES = 100;

        for (num_init_tries = 1; num_init_tries <= MAX_INIT_TRIES;
             ++num_init_tries) {
          for (int i = 0; i < cont_params.size(); ++i)
            cont_params(i) = init_rng();
          if (initialize_state_values(cont_params, model, writer))
            break;
        }

        if (num_init_tries > MAX_INIT_TRIES) {
          std::stringstream R_ss, MAX_INIT_TRIES_ss;
          R_ss << R;
          MAX_INIT_TRIES_ss << MAX_INIT_TRIES;

          writer();
          writer();
          writer("Initialization between (-" + R_ss.str() + ", " + R_ss.str()
                 + ") failed after "
                 + MAX_INIT_TRIES_ss.str() + " attempts. ");
          writer(" Try specifying initial values,"
                 " reducing ranges of constrained values,"
                 " or reparameterizing the model.");
          return false;
        }
        return true;
      }


      /**
       * Creates the initial state.
       *
       * @param[in]     source      a string that the context_factory can
       *                            interpret and provide a valid var_context
       * @param[in]     R           a double to specify the range of
       *                            random inits
       * @param[out]    cont_params the initialized state. This should be the
       *                            right size and set to 0.
       * @param[in,out] model       the model. Side effects on model? I'm not
       *                            quite sure
       * @param[in,out] base_rng    the random number generator.
       *                            State may change.
       * @param[in,out] writer      writer callback for messages
       * @param[in,out] context_factory  an instantiated factory that implements
       *                            the concept of a context_factory. This has
       *                            one method that takes a string.
       */
      template <class ContextFactory, class Model, class RNG>
      bool initialize_state_source_and_random(const std::string& source,
                                              double R,
                                              Eigen::VectorXd& cont_params,
                                              Model& model,
                                              RNG& base_rng,
                               interface_callbacks::writer::base_writer& writer,
                                              ContextFactory& context_factory) {
        try {
          boost::random::uniform_real_distribution<double>
            init_range_distribution(-R, R);
          boost::variate_generator
            <RNG&, boost::random::uniform_real_distribution<double> >
            init_rng(base_rng, init_range_distribution);

          static int MAX_INIT_TRIES = 100;

          int num_init_tries = -1;
          std::vector<std::string> cont_names;
          model.constrained_param_names(cont_names, false, false);
          rm_indices_from_name(cont_names);
          for (num_init_tries = 1; num_init_tries <= MAX_INIT_TRIES;
               ++num_init_tries) {
            std::vector<double> cont_vecs(cont_params.size());
            for (int i = 0; i < cont_params.size(); ++i) {
              cont_vecs[i] = init_rng();
              cont_params[i] = cont_vecs[i];
            }
            typename ContextFactory::var_context_t context
              = context_factory(source);
            std::vector<double> cont_vecs_constrained;
            std::vector<int> int_vecs;
            model.write_array(base_rng, cont_vecs, int_vecs,
                              cont_vecs_constrained, false, false, 0);
            std::vector<std::vector<size_t> > dims;
            model.get_dims(dims);
            stan::io::array_var_context random_context(cont_names,
                                                       cont_vecs_constrained,
                                                       dims);
            stan::io::chained_var_context cvc(context, random_context);
            model.transform_inits(cvc, cont_params, 0);
            if (initialize_state_values(cont_params, model, writer))
              break;
          }

          if (num_init_tries > MAX_INIT_TRIES) {
            std::stringstream R_ss, MAX_INIT_TRIES_ss;
            R_ss << R;
            MAX_INIT_TRIES_ss << MAX_INIT_TRIES;

            writer();
            writer();
            writer("Initialization between (-" + R_ss.str() + ", " + R_ss.str()
                   + ") failed after "
                   + MAX_INIT_TRIES_ss.str() + " attempts. ");
            writer(" Try specifying initial values,"
                   " reducing ranges of constrained values,"
                   " or reparameterizing the model.");
            return false;
          }
        } catch(const std::exception& e) {
          writer("Initialization partially from source failed.");
          writer(e.what());
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
       * @param[in,out] base_rng    the random number generator.
       *                            State may change.
       * @param[in,out] writer      writer callback for messages
       * @param[in,out] context_factory  an instantiated factory that implements
       *                            the concept of a context_factory. This has
       *                            one method that takes a string.
       * @param[in] enable_random_init true or false
       * @param[in] R               a double for the range of generating
       *                            random inits. it's used for randomly
       *                            generating partial inits
       */
      template <class ContextFactory, class Model, class RNG>
      bool initialize_state_source(const std::string source,
                                   Eigen::VectorXd& cont_params,
                                   Model& model,
                                   RNG& base_rng,
                                   interface_callbacks::writer::base_writer&
                                   writer,
                                   ContextFactory& context_factory,
                                   bool enable_random_init = false,
                                   double R = 2) {
        try {
          typename ContextFactory::var_context_t context
            = context_factory(source);

          if (enable_random_init && !are_all_pars_initialized(model, context)) {
            return initialize_state_source_and_random(source,
                                                      R,
                                                      cont_params,
                                                      model,
                                                      base_rng,
                                                      writer,
                                                      context_factory);
          }
          model.transform_inits(context, cont_params, 0);
        } catch(const std::exception& e) {
          writer("Initialization from source failed.");
          writer(e.what());
          return false;
        }
        return initialize_state_values(cont_params, model, writer);
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
       * @param[in]     init        init can either be "0", a number as a
       *                            string, or a filename.
       * @param[out]    cont_params the initialized state. This should be the
       *                            right size and set to 0.
       * @param[in,out] model       the model. Side effects on model? I'm not
       *                            quite sure
       * @param[in,out] base_rng    the random number generator.
       *                            State may change.
       * @param[in,out] writer      writer callback for messages
       * @param[in,out] context_factory  an instantiated factory that implements
       *                            the concept of a context_factory. This has
       *                            one method that takes a string.
       * @param[in] enable_random_init true or false.
       * @param[in] init_r          a double for the range of generating
       *                            random inits. it's used for randomly
       *                            generating partial inits
       */
      template <class ContextFactory, class Model, class RNG>
      bool initialize_state(const std::string& init,
                            Eigen::VectorXd& cont_params,
                            Model& model,
                            RNG& base_rng,
                            interface_callbacks::writer::base_writer& writer,
                            ContextFactory& context_factory,
                            bool enable_random_init = false,
                            double init_r = 2) {
        double R;
        if (get_double_from_string(init, R)) {
          if (R == 0) {
            return initialize_state_zero(cont_params, model, writer);
          } else {
            return initialize_state_random(R, cont_params, model,
                                           base_rng, writer);
          }
        }
        return initialize_state_source(init, cont_params, model,
                                       base_rng, writer,
                                       context_factory,
                                       enable_random_init,
                                       init_r);
      }

    }  // init
  }  // services
}  // stan
#endif
