#ifndef STAN_SERVICES_OPTIMIZE_BFGS_HPP
#define STAN_SERVICES_OPTIMIZE_BFGS_HPP

#include <stan/interface_callbacks/writer/base_writer.hpp>
#include <stan/optimization/bfgs.hpp>
#include <stan/services/error_codes.hpp>
#include <stan/services/io/do_print.hpp>
#include <stan/services/io/write_iteration.hpp>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

namespace stan {
  namespace services {
    namespace optimize {

      template<typename Model, typename RNGT,
               typename StartIterationCallback>
      int bfgs(Model &model,
               RNGT &base_rng,
               Eigen::VectorXd &cont_params,
               double init_alpha,
               double tol_obj,
               double tol_rel_obj,
               double tol_grad,
               double tol_rel_grad,
               double tol_param,
               int num_iterations,
               bool save_iterations,
               int refresh,
               StartIterationCallback& interrupt,
               interface_callbacks::writer::base_writer& message_writer,
               interface_callbacks::writer::base_writer& parameter_writer) {
        std::vector<double> cont_vector(cont_params.size());
        for (int i = 0; i < cont_params.size(); ++i)
          cont_vector.at(i) = cont_params(i);
        std::vector<int> disc_vector;

        std::stringstream bfgs_ss;
        
        typedef stan::optimization::BFGSLineSearch
          <Model,stan::optimization::BFGSUpdate_HInv<> > Optimizer;
        Optimizer bfgs(model, cont_vector, disc_vector, &bfgs_ss);
        bfgs._ls_opts.alpha0 = init_alpha;
        bfgs._conv_opts.tolAbsF = tol_obj;
        bfgs._conv_opts.tolRelF = tol_rel_obj;
        bfgs._conv_opts.tolAbsGrad = tol_grad;
        bfgs._conv_opts.tolRelGrad = tol_rel_grad;
        bfgs._conv_opts.tolAbsX = tol_param;
        bfgs._conv_opts.maxIts = num_iterations;
        
        double lp = bfgs.logp();
        
        std::stringstream msg;
        msg << "Initial log joint probability = " << lp;
        message_writer(msg.str());

        std::vector<std::string> names;
        names.push_back("lp__");
        model.constrained_param_names(names, true, true);
        parameter_writer(names);
        
        if (save_iterations) {
          io::write_iteration(model, base_rng,
                              lp, cont_vector, disc_vector,
                              message_writer, parameter_writer);
        }

        int ret = 0;

        while (ret == 0) {
          interrupt();
          if (io::do_print(bfgs.iter_num(), 50*refresh)) {
            message_writer("    Iter "
                           "     log prob "
                           "       ||dx|| "
                           "     ||grad|| "
                           "      alpha "
                           "     alpha0 "
                           " # evals "
                           " Notes ");
          }

          ret = bfgs.step();
          lp = bfgs.logp();
          bfgs.params_r(cont_vector);

          if (io::do_print(bfgs.iter_num(),
                           ret != 0 || !bfgs.note().empty(), refresh)) {
            msg.str("");
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
            io::write_iteration(model, base_rng,
                                lp, cont_vector, disc_vector,
                                message_writer, parameter_writer);
          }
        }

        if (!save_iterations)
          io::write_iteration(model, base_rng,
                              lp, cont_vector, disc_vector,
                              message_writer, parameter_writer);

        for (int i = 0; i < cont_params.size(); ++i)
          cont_params[i] = cont_vector[i];
        
        int return_code;
        if (ret >= 0) {
          message_writer("Optimization terminated normally: ");
          return_code = stan::services::error_codes::OK;
        } else {
          message_writer("Optimization terminated with error: ");
          return_code = stan::services::error_codes::SOFTWARE;
        }
        message_writer("  " + bfgs.get_code_string(ret));

        return return_code;
      }

    }
  }
}
#endif
