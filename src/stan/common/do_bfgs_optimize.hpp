#ifndef STAN__COMMON__DO_BFGS_OPTIMIZE_HPP
#define STAN__COMMON__DO_BFGS_OPTIMIZE_HPP

#include <fstream>
#include <iostream>
#include <iomanip>
#include <vector>

#include <stan/gm/error_codes.hpp>

#include <stan/common/do_print.hpp>
#include <stan/common/write_iteration_csv.hpp>
#include <stan/common/write_iteration.hpp>

namespace stan {

  namespace common {

    template<typename ModelT,typename BFGSOptimizerT,typename RNGT,
             typename StartIterationCallback>
    int do_bfgs_optimize(ModelT &model, BFGSOptimizerT &bfgs,
                         RNGT &base_rng,
                         double &lp,
                         std::vector<double> &cont_vector,
                         std::vector<int> &disc_vector,
                         std::fstream* output_stream,
                         std::ostream* notice_stream,
                         bool save_iterations,
                         int refresh,
                         StartIterationCallback& callback) {
      lp = bfgs.logp();
          
      if (notice_stream) 
        (*notice_stream) << "initial log joint probability = " << lp << std::endl;
      if (output_stream && save_iterations) {
        write_iteration(*output_stream, model, base_rng,
                        lp, cont_vector, disc_vector);
      }

      int ret = 0;
          
      while (ret == 0) {  
        callback();
        if (notice_stream && do_print(bfgs.iter_num(), 50*refresh)) {
          (*notice_stream) << "    Iter ";
          (*notice_stream) << "     log prob ";
          (*notice_stream) << "       ||dx|| ";
          (*notice_stream) << "     ||grad|| ";
          (*notice_stream) << "      alpha ";
          (*notice_stream) << "     alpha0 ";
          (*notice_stream) << " # evals ";
          (*notice_stream) << " Notes " << std::endl;
        }
            
        ret = bfgs.step();
        lp = bfgs.logp();
        bfgs.params_r(cont_vector);
            
        if (notice_stream && (do_print(bfgs.iter_num(), ret != 0 || !bfgs.note().empty(),refresh))) {
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
        }
            
        if (output_stream && save_iterations) {
          write_iteration(*output_stream, model, base_rng,
                          lp, cont_vector, disc_vector);
        }
      }
          
      int return_code;
      if (ret >= 0) {
        if (notice_stream)
          (*notice_stream) << "Optimization terminated normally: " << std::endl;
        return_code = stan::gm::error_codes::OK;
      } else {
        if (notice_stream)
          (*notice_stream) << "Optimization terminated with error: " << std::endl;
        return_code = stan::gm::error_codes::SOFTWARE;
      }
      if (notice_stream)
        (*notice_stream) << "  " << bfgs.get_code_string(ret) << std::endl;

      return return_code;
    }

  }

}
#endif

