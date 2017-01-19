#ifndef STAN_SERVICES_OPTIMIZE_BFGS_HPP
#define STAN_SERVICES_OPTIMIZE_BFGS_HPP

#include <stan/io/var_context.hpp>
#include <stan/io/chained_var_context.hpp>
#include <stan/callbacks/interrupt.hpp>
#include <stan/io/random_var_context.hpp>
#include <stan/callbacks/writer.hpp>
#include <stan/services/error_codes.hpp>
#include <stan/optimization/bfgs.hpp>
#include <stan/services/util/initialize.hpp>
#include <stan/services/util/create_rng.hpp>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

namespace stan {
  namespace services {
    namespace optimize {

      /**
       * Runs the BFGS algorithm for a model.
       *
       * @tparam Model A model implementation
       * @param[in] model Input model to test (with data already instantiated)
       * @param[in] init var context for initialization
       * @param[in] random_seed random seed for the random number generator
       * @param[in] chain chain id to advance the pseudo random number generator
       * @param[in] init_radius radius to initialize
       * @param[in] init_alpha line search step size for first iteration
       * @param[in] tol_obj convergence tolerance on absolute changes in
       *   objective function value
       * @param[in] tol_rel_obj convergence tolerance on relative changes
       *   in objective function value
       * @param[in] tol_grad convergence tolerance on the norm of the gradient
       * @param[in] tol_rel_grad convergence tolerance on the relative norm of
       *   the gradient
       * @param[in] tol_param convergence tolerance on changes in parameter
       *   value
       * @param[in] num_iterations maximum number of iterations
       * @param[in] save_iterations indicates whether all the interations should
       *   be saved to the parameter_writer
       * @param[in] refresh how often to write output to message_writer
       * @param[in,out] interrupt callback to be called every iteration
       * @param[in,out] message_writer output for messages
       * @param[in,out] init_writer Writer callback for unconstrained inits
       * @param[in,out] parameter_writer output for parameter values
       * @return error_codes::OK if successful
       */
      template <class Model>
      int bfgs(Model& model, stan::io::var_context& init,
               unsigned int random_seed, unsigned int chain,
               double init_radius, double init_alpha, double tol_obj,
               double tol_rel_obj, double tol_grad, double tol_rel_grad,
               double tol_param, int num_iterations, bool save_iterations,
               int refresh,
               callbacks::interrupt& interrupt,
               callbacks::writer& message_writer,
               callbacks::writer& init_writer,
               callbacks::writer& parameter_writer) {
        boost::ecuyer1988 rng = util::create_rng(random_seed, chain);

        std::vector<int> disc_vector;
        std::vector<double> cont_vector
          = util::initialize(model, init, rng, init_radius, false,
                             message_writer, init_writer);

        std::stringstream bfgs_ss;
        typedef stan::optimization::BFGSLineSearch
          <Model, stan::optimization::BFGSUpdate_HInv<> > Optimizer;
        Optimizer bfgs(model, cont_vector, disc_vector, &bfgs_ss);
        bfgs.ls_opts_.alpha0 = init_alpha;
        bfgs.conv_opts_.tolAbsF = tol_obj;
        bfgs.conv_opts_.tolRelF = tol_rel_obj;
        bfgs.conv_opts_.tolAbsGrad = tol_grad;
        bfgs.conv_opts_.tolRelGrad = tol_rel_grad;
        bfgs.conv_opts_.tolAbsX = tol_param;
        bfgs.conv_opts_.maxIts = num_iterations;

        double lp = bfgs.logp();

        std::stringstream initial_msg;
        initial_msg << "Initial log joint probability = " << lp;
        message_writer(initial_msg.str());

        std::vector<std::string> names;
        names.push_back("lp__");
        model.constrained_param_names(names, true, true);
        parameter_writer(names);

        if (save_iterations) {
          std::vector<double> values;
          std::stringstream msg;
          model.write_array(rng, cont_vector, disc_vector, values,
                            true, true, &msg);
          if (msg.str().length() > 0)
            message_writer(msg.str());

          values.insert(values.begin(), lp);
          parameter_writer(values);
        }
        int ret = 0;

        while (ret == 0) {
          interrupt();
          if (refresh > 0
              && (bfgs.iter_num() == 0
                  || ((bfgs.iter_num() + 1) % refresh == 0)))
            message_writer("    Iter"
                           "      log prob"
                           "        ||dx||"
                           "      ||grad||"
                           "       alpha"
                           "      alpha0"
                           "  # evals"
                           "  Notes ");

          ret = bfgs.step();
          lp = bfgs.logp();
          bfgs.params_r(cont_vector);

          if (refresh > 0
              && (ret != 0
                  || !bfgs.note().empty()
                  || bfgs.iter_num() == 0
                  || ((bfgs.iter_num() + 1) % refresh == 0))) {
            std::stringstream msg;
            msg << " " << std::setw(7) << bfgs.iter_num() << " ";
            msg << " " << std::setw(12) << std::setprecision(6)
                << lp << " ";
            msg << " " << std::setw(12) << std::setprecision(6)
                << bfgs.prev_step_size() << " ";
            msg << " " << std::setw(12) << std::setprecision(6)
                << bfgs.curr_g().norm() << " ";
            msg << " " << std::setw(10) << std::setprecision(4)
                << bfgs.alpha() << " ";
            msg << " " << std::setw(10) << std::setprecision(4)
                << bfgs.alpha0() << " ";
            msg << " " << std::setw(7)
                << bfgs.grad_evals() << " ";
            msg << " " << bfgs.note() << " ";
            message_writer(msg.str());
          }

          if (bfgs_ss.str().length() > 0) {
            message_writer(bfgs_ss.str());
            bfgs_ss.str("");
          }

          if (save_iterations) {
            std::vector<double> values;
            std::stringstream msg;
            model.write_array(rng, cont_vector, disc_vector, values,
                              true, true, &msg);
            // This if is here to match the pre-refactor behavior
            if (msg.str().length() > 0)
              message_writer(msg.str());

            values.insert(values.begin(), lp);
            parameter_writer(values);
          }
        }

        if (!save_iterations) {
          std::vector<double> values;
          std::stringstream msg;
          model.write_array(rng, cont_vector, disc_vector, values,
                            true, true, &msg);
          if (msg.str().length() > 0)
            message_writer(msg.str());
          values.insert(values.begin(), lp);
          parameter_writer(values);
        }

        int return_code;
        if (ret >= 0) {
          message_writer("Optimization terminated normally: ");
          return_code = error_codes::OK;
        } else {
          message_writer("Optimization terminated with error: ");
          return_code = error_codes::SOFTWARE;
        }
        message_writer("  " + bfgs.get_code_string(ret));

        return return_code;
      }

    }
  }
}
#endif
