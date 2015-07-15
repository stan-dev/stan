#ifndef STAN_SERVICES_OPTIMIZE_OPTIMIZE_HPP
#define STAN_SERVICES_OPTIMIZE_OPTIMIZE_HPP

#include <cmath>
#include <sstream>
#include <iomanip>

#include <stan/services/arguments/arg_adapt.hpp>
#include <stan/services/arguments/arg_adapt_delta.hpp>
#include <stan/services/arguments/arg_adapt_engaged.hpp>
#include <stan/services/arguments/arg_adapt_gamma.hpp>
#include <stan/services/arguments/arg_adapt_init_buffer.hpp>
#include <stan/services/arguments/arg_adapt_kappa.hpp>
#include <stan/services/arguments/arg_adapt_t0.hpp>
#include <stan/services/arguments/arg_adapt_term_buffer.hpp>
#include <stan/services/arguments/arg_adapt_window.hpp>
#include <stan/services/arguments/arg_bfgs.hpp>
#include <stan/services/arguments/arg_data.hpp>
#include <stan/services/arguments/arg_data_file.hpp>
#include <stan/services/arguments/arg_dense_e.hpp>
#include <stan/services/arguments/arg_diag_e.hpp>
#include <stan/services/arguments/arg_diagnose.hpp>
#include <stan/services/arguments/arg_diagnostic_file.hpp>
#include <stan/services/arguments/arg_engine.hpp>
#include <stan/services/arguments/arg_fail.hpp>
#include <stan/services/arguments/arg_fixed_param.hpp>
#include <stan/services/arguments/arg_history_size.hpp>
#include <stan/services/arguments/arg_hmc.hpp>
#include <stan/services/arguments/arg_id.hpp>
#include <stan/services/arguments/arg_init.hpp>
#include <stan/services/arguments/arg_init_alpha.hpp>
#include <stan/services/arguments/arg_int_time.hpp>
#include <stan/services/arguments/arg_iter.hpp>
#include <stan/services/arguments/arg_lbfgs.hpp>
#include <stan/services/arguments/arg_max_depth.hpp>
#include <stan/services/arguments/arg_method.hpp>
#include <stan/services/arguments/arg_metric.hpp>
#include <stan/services/arguments/arg_newton.hpp>
#include <stan/services/arguments/arg_num_samples.hpp>
#include <stan/services/arguments/arg_num_warmup.hpp>
#include <stan/services/arguments/arg_nuts.hpp>
#include <stan/services/arguments/arg_optimize.hpp>
#include <stan/services/arguments/arg_optimize_algo.hpp>
#include <stan/services/arguments/arg_output.hpp>
#include <stan/services/arguments/arg_output_file.hpp>
#include <stan/services/arguments/arg_random.hpp>
#include <stan/services/arguments/arg_refresh.hpp>
#include <stan/services/arguments/arg_rwm.hpp>
#include <stan/services/arguments/arg_sample.hpp>
#include <stan/services/arguments/arg_sample_algo.hpp>
#include <stan/services/arguments/arg_save_iterations.hpp>
#include <stan/services/arguments/arg_save_warmup.hpp>
#include <stan/services/arguments/arg_seed.hpp>
#include <stan/services/arguments/arg_static.hpp>
#include <stan/services/arguments/arg_stepsize.hpp>
#include <stan/services/arguments/arg_stepsize_jitter.hpp>
#include <stan/services/arguments/arg_test.hpp>
#include <stan/services/arguments/arg_test_grad_eps.hpp>
#include <stan/services/arguments/arg_test_grad_err.hpp>
#include <stan/services/arguments/arg_test_gradient.hpp>
#include <stan/services/arguments/arg_thin.hpp>
#include <stan/services/arguments/arg_tolerance.hpp>
#include <stan/services/arguments/arg_unit_e.hpp>
#include <stan/services/arguments/argument.hpp>
#include <stan/services/arguments/argument_parser.hpp>
#include <stan/services/arguments/argument_probe.hpp>
#include <stan/services/arguments/categorical_argument.hpp>
#include <stan/services/arguments/list_argument.hpp>
#include <stan/services/arguments/singleton_argument.hpp>
#include <stan/services/arguments/unvalued_argument.hpp>
#include <stan/services/arguments/valued_argument.hpp>

#include <stan/services/init/initialize_state.hpp>

#include <stan/services/error_codes.hpp>

#include <stan/services/optimize/do_bfgs_optimize.hpp>

namespace stan {
  namespace services {
    namespace optimize {

      template <class Model, class RNG,
                class InfoWriter, class ErrWriter,
                class OutputWriter, class Interrupt>
      int optimize(Eigen::VectorXd& cont_params,
                   Model& model,
                   RNG& base_rng,
                   stan::services::categorical_argument* optimize_args,
                   int refresh,
                   InfoWriter& info,
                   ErrWriter& err,
                   OutputWriter& output,
                   Interrupt& iteration_interrupt) {

        std::stringstream msg;

        std::vector<double> cont_vector(cont_params.size());
        for (int i = 0; i < cont_params.size(); ++i)
          cont_vector.at(i) = cont_params(i);
        std::vector<int> disc_vector;

        stan::services::list_argument* algo
          = dynamic_cast<stan::services::list_argument*>
            (optimize_args->arg("algorithm"));

        int num_iterations
          = dynamic_cast<stan::services::int_argument*>
            (optimize_args->arg("iter"))->value();

        bool save_iterations
          = dynamic_cast<stan::services::bool_argument*>
            (optimize_args->arg("save_iterations"))->value();

        std::vector<std::string> names;
        names.push_back("lp__");
        model.constrained_param_names(names, true, true);
        output(names);

        double lp(0);

        if (algo->value() == "newton") {

          std::vector<double> gradient;
          try {
            msg.str(std::string());
            msg.clear();
            lp = model.template log_prob<false, false>
                   (cont_vector, disc_vector, &msg);
            if (msg.str().size()) info(msg.str());
          } catch (const std::exception& e) {
            services::io::write_error_msg(err, e);
            lp = -std::numeric_limits<double>::infinity();
          }

          msg.str(std::string());
          msg.clear();
          msg << "initial log joint probability = " << lp;
          info(msg.str());

          if (save_iterations) {
            services::io::write_iteration(output, model, base_rng,
                                          lp, cont_vector, disc_vector);
          }

          double lastlp = lp * 1.1;
          int m = 0;

          msg.str(std::string());
          msg.clear();
          msg << "(lp - lastlp) / lp > 1e-8: " << (lp - lastlp) / std::fabs(lp);
          info(msg.str());

          while ((lp - lastlp) / std::fabs(lp) > 1e-8) {
            lastlp = lp;
            lp = stan::optimization::newton_step
            (model, cont_vector, disc_vector);

            msg.str(std::string());
            msg.clear();
            msg << "Iteration "
                << std::setw(2) << (m + 1) << ". "
                << "Log joint probability = " << std::setw(10) << lp
                << ". Improved by " << (lp - lastlp) << ".";
            info(msg.str());

            m++;

            if (save_iterations) {
              io::write_iteration(output, model, base_rng,
                                  lp, cont_vector, disc_vector);
            }
          }

          return stan::services::error_codes::OK;

        }

        if (algo->value() == "bfgs") {

          msg.str(std::string());
          msg.clear();

          typedef stan::optimization::BFGSLineSearch
            <Model,stan::optimization::BFGSUpdate_HInv<> > Optimizer;
          Optimizer bfgs(model, cont_vector, disc_vector, &msg);

          bfgs._ls_opts.alpha0
            = dynamic_cast<stan::services::real_argument*>
              (algo->arg("bfgs")->arg("init_alpha"))->value();
          bfgs._conv_opts.tolAbsF
            = dynamic_cast<stan::services::real_argument*>
              (algo->arg("bfgs")->arg("tol_obj"))->value();
          bfgs._conv_opts.tolRelF
            = dynamic_cast<stan::services::real_argument*>
              (algo->arg("bfgs")->arg("tol_rel_obj"))->value();
          bfgs._conv_opts.tolAbsGrad
            = dynamic_cast<stan::services::real_argument*>
              (algo->arg("bfgs")->arg("tol_grad"))->value();
          bfgs._conv_opts.tolRelGrad
            = dynamic_cast<stan::services::real_argument*>
              (algo->arg("bfgs")->arg("tol_rel_grad"))->value();
          bfgs._conv_opts.tolAbsX
            = dynamic_cast<stan::services::real_argument*>
              (algo->arg("bfgs")->arg("tol_param"))->value();
          bfgs._conv_opts.maxIts = num_iterations;

          return do_bfgs_optimize(model,bfgs, base_rng,
                                  lp, cont_vector, disc_vector,
                                  output, info,
                                  save_iterations, refresh,
                                  iteration_interrupt);

          if (msg.str().size()) info(msg.str());

        }

        if (algo->value() == "lbfgs") {

          msg.str(std::string());
          msg.clear();

          typedef stan::optimization::BFGSLineSearch
            <Model,stan::optimization::LBFGSUpdate<> > Optimizer;
          Optimizer bfgs(model, cont_vector, disc_vector, &msg);

          bfgs.get_qnupdate().set_history_size(dynamic_cast<services::int_argument*>
                                               (algo->arg("lbfgs")->arg("history_size"))->value());
          bfgs._ls_opts.alpha0
            = dynamic_cast<services::real_argument*>
              (algo->arg("lbfgs")->arg("init_alpha"))->value();
          bfgs._conv_opts.tolAbsF
            = dynamic_cast<services::real_argument*>
              (algo->arg("lbfgs")->arg("tol_obj"))->value();
          bfgs._conv_opts.tolRelF
            = dynamic_cast<services::real_argument*>
              (algo->arg("lbfgs")->arg("tol_rel_obj"))->value();
          bfgs._conv_opts.tolAbsGrad
            = dynamic_cast<services::real_argument*>
              (algo->arg("lbfgs")->arg("tol_grad"))->value();
          bfgs._conv_opts.tolRelGrad
            = dynamic_cast<services::real_argument*>
              (algo->arg("lbfgs")->arg("tol_rel_grad"))->value();
          bfgs._conv_opts.tolAbsX
            = dynamic_cast<services::real_argument*>
              (algo->arg("lbfgs")->arg("tol_param"))->value();
          bfgs._conv_opts.maxIts = num_iterations;

          return do_bfgs_optimize(model,bfgs, base_rng,
                                  lp, cont_vector, disc_vector,
                                  output, info,
                                  save_iterations, refresh,
                                  iteration_interrupt);

          if (msg.str().size()) info(msg.str());

        }

        return stan::services::error_codes::USAGE;

      }

    } // sample
  } // services
} // stan

#endif
