#ifndef STAN__SERVICES__OPTIMIZATION__DO_BFGS_OPTIMIZE_HPP
#define STAN__SERVICES__OPTIMIZATION__DO_BFGS_OPTIMIZE_HPP

#include <fstream>
#include <iostream>
#include <iomanip>
#include <vector>

#include <stan/services/error_codes.hpp>
#include <stan/services/io.hpp>

namespace stan {
  namespace services {
    namespace optimization {

      template<typename ModelT,typename BFGSOptimizerT,typename RNGT,
               typename OutputWriter, typename InfoWriter, typename Interrupt>
      int do_bfgs_optimize(ModelT &model,
                           BFGSOptimizerT &bfgs,
                           RNGT &base_rng,
                           double &lp,
                           std::vector<double> &cont_vector,
                           std::vector<int> &disc_vector,
                           OutputWriter& output_stream,
                           InfoWriter& info,
                           bool save_iterations,
                           int refresh,
                           Interrupt& interrupt) {
        lp = bfgs.logp();
        
        info.write_message("initial log joint probability = "
                           + InfoWriter::to_string(lp));
        
        if (save_iterations) {
          io::write_iteration(output_stream, model, base_rng,
                              lp, cont_vector, disc_vector);
        }

        int ret = 0;
            
        while (ret == 0) {  
          interrupt();
          if (io::do_print(bfgs.iter_num(), 50 * refresh)) {
            info.write_message(std::string("    Iter ")
                               + "     log prob "
                               + "       ||dx|| "
                               + "     ||grad|| "
                               + "      alpha "
                               + "     alpha0 "
                               + " # evals "
                               + " Notes ");
          }
              
          ret = bfgs.step();
          lp = bfgs.logp();
          bfgs.params_r(cont_vector);
              
          if (io::do_print(bfgs.iter_num(), ret != 0 || !bfgs.note().empty(),refresh)) {
            info.write_message(" "
                               + InfoWriter::to_string(bfgs.iter_num(), 7) + " " + " "
                               + InfoWriter::to_string(lp, 12) + " " + " "
                               + InfoWriter::to_string(bfgs.prev_step_size(), 12) + " " + " "
                               + InfoWriter::to_string(bfgs.curr_g().norm(), 12) + " " + " "
                               + InfoWriter::to_string(bfgs.alpha(), 10) + " " + " "
                               + InfoWriter::to_string(bfgs.alpha0(), 10) + " " + " "
                               + InfoWriter::to_string(bfgs.grad_evals(), 7) +" " + " "
                               + bfgs.note() + " ");
            
            /*
             (*notice_stream) << " " << std::setw(7) << bfgs.iter_num() << " ";
             (*notice_stream) << " " << std::setw(12) << std::setprecision(6)
             << lp << " ";
             (*notice_stream) << " " << std::setw(12) << std::setprecision(6)
             << bfgs.prev_step_size() << " ";
             (*notice_stream) << " " << std::setw(12) << std::setprecision(6)
             << bfgs.curr_g().norm() << " ";
             (*notice_stream) << " " << std::setw(10) << std::setprecision(4)
             << bfgs.alpha() << " ";
             (*notice_stream) << " " << std::setw(10) << std::setprecision(4)
             << bfgs.alpha0() << " ";
             (*notice_stream) << " " << std::setw(7)
             << bfgs.grad_evals() << " ";
             (*notice_stream) << " " << bfgs.note() << " ";
             (*notice_stream) << std::endl;
            */
          }
             
          if (save_iterations) {
            io::write_iteration(output_stream, model, base_rng,
                               lp, cont_vector, disc_vector);
          }
        }
            
        int return_code;
        if (ret >= 0) {
          info.write_message("Optimization terminated normally: ");
          return_code = stan::services::error_codes::OK;
        } else {
          info.write_message("Optimization terminated with error: ");
          return_code = stan::services::error_codes::SOFTWARE;
        }
        info.write_message("  " + bfgs.get_code_string(ret));

        return return_code;
      }
      
    }
  }
}
#endif

