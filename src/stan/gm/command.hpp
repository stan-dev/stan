#ifndef __STAN__GM__COMMAND_HPP__
#define __STAN__GM__COMMAND_HPP__

#include <fstream>
#include <stdexcept>

#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <boost/random/additive_combine.hpp> // L'Ecuyer RNG
#include <boost/random/uniform_real_distribution.hpp>

#include <stan/version.hpp>
#include <stan/io/cmd_line.hpp>
#include <stan/io/dump.hpp>

#include <stan/gm/arguments/argument_parser.hpp>
#include <stan/gm/arguments/arg_id.hpp>
#include <stan/gm/arguments/arg_data.hpp>
#include <stan/gm/arguments/arg_random.hpp>
#include <stan/gm/arguments/arg_method.hpp>
#include <stan/gm/arguments/arg_output.hpp>

#include <stan/mcmc/hmc/static/adapt_unit_e_static_hmc.hpp>
#include <stan/mcmc/hmc/static/adapt_diag_e_static_hmc.hpp>
#include <stan/mcmc/hmc/static/adapt_dense_e_static_hmc.hpp>
#include <stan/mcmc/hmc/nuts/adapt_unit_e_nuts.hpp>
#include <stan/mcmc/hmc/nuts/adapt_diag_e_nuts.hpp>
#include <stan/mcmc/hmc/nuts/adapt_dense_e_nuts.hpp>

#include <stan/optimization/newton.hpp>
#include <stan/optimization/nesterov_gradient.hpp>
#include <stan/optimization/bfgs.hpp>

namespace stan {

  namespace gm {
    
    void write_error_msg(std::ostream* error_stream,
                         const std::domain_error& e) {
      
      if (!error_stream) return;
      
      *error_stream << std::endl
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
    
    bool do_print(int n, int refresh) {
      return (refresh > 0) &&
      (n == 0 || ((n + 1) % refresh == 0) );
    }

    void print_progress(int m, int num_iterations, int refresh, bool warmup) {
      
      int it_print_width = std::ceil(std::log10(num_iterations));
      
      if (do_print(m, refresh)) {
        
        std::cout << "Iteration: ";
        std::cout << std::setw(it_print_width) << (m + 1)
                  << " / " << num_iterations;
          
        std::cout << " [" << std::setw(3) 
                  << static_cast<int>( (100.0 * (m + 1)) / num_iterations )
                  << "%] ";
        std::cout << (warmup ? " (Warmup)" : " (Sampling)");
        std::cout << std::endl;
          
      }
    
    }

    template <class Sampler, class Model, class RNG>
    void print_sample(std::ostream& sample_file_stream,
                      std::ostream& debug_file_stream,
                      stan::mcmc::sample& s, 
                      Sampler& sampler,
                      Model& model,
                      RNG& base_rng) {
      
      // Temporary as model::write_csv isn't a const method
      std::vector<double> cont(s.cont_params());
      std::vector<int> disc(s.disc_params());
      
      sample_file_stream << s.log_prob() << ",";
      sampler.write_sampler_params(sample_file_stream);
      model.write_csv(base_rng, cont, disc, 
                      sample_file_stream, &std::cout);
      
      //sampler.z().write(debug_file_stream);
      //debug_file_stream << std::endl;
      
      
    }
    
    template <class Sampler, class Model, class RNG>
    void run_markov_chain(Sampler& sampler,
                          int num_iterations,
                          int num_thin,
                          int refresh,
                          bool save,
                          bool warmup,
                          std::ostream& sample_file_stream,
                          std::ostream& debug_file_stream,
                          stan::mcmc::sample& init_s,
                          Model& model,
                          RNG& base_rng) {
      
      for (size_t m = 0; m < num_iterations; ++m) {
      
        print_progress(m, num_iterations, refresh, warmup);
      
        init_s = sampler.transition(init_s);
          
        if ( save && ( (m % num_thin) == 0) ) {
          print_sample<Sampler, Model, RNG>(sample_file_stream, debug_file_stream,
                                            init_s, sampler, model, base_rng);
        }

      }
      
    }

    template <class Sampler, class Model, class RNG>
    void warmup(Sampler& sampler,
                int num_iterations,
                int num_thin,
                int refresh,
                bool save,
                std::ostream& sample_file_stream,
                std::ostream& debug_file_stream,
                stan::mcmc::sample& init_s,
                Model& model,
                RNG& base_rng) {
      
      run_markov_chain<Sampler, Model, RNG>(sampler, num_iterations, num_thin, 
                                            refresh, save, true,
                                            sample_file_stream,
                                            debug_file_stream,
                                            init_s, model, base_rng);
      
    }

    template <class Sampler, class Model, class RNG>
    void sample(Sampler& sampler,
                int num_iterations,
                int num_thin,
                int refresh,
                bool save,
                std::ostream& sample_file_stream,
                std::ostream& debug_file_stream,
                stan::mcmc::sample& init_s,
                Model& model,
                RNG& base_rng) {
      
      run_markov_chain<Sampler, Model, RNG>(sampler, num_iterations, num_thin, 
                                            refresh, save, false,
                                            sample_file_stream,
                                            debug_file_stream,
                                            init_s, model, base_rng);
      
    }
    
    template <class Model>
    int nuts_command(int argc, const char* argv[]) {

      std::vector<argument*> valid_arguments;
      valid_arguments.push_back(new arg_id());
      valid_arguments.push_back(new arg_data());
      valid_arguments.push_back(new arg_random());
      valid_arguments.push_back(new arg_output());
      valid_arguments.push_back(new arg_method());
      
      argument_parser parser(valid_arguments);
      
      parser.parse_args(argc, argv, &std::cout);

      // Identification
      unsigned int id = dynamic_cast<int_argument*>(parser.arg("id"))->value();
      
      //////////////////////////////////////////////////
      //            Random number generator           //
      //////////////////////////////////////////////////
      
      unsigned int random_seed = dynamic_cast<int_argument*>(
                                 parser.arg("random")->arg("seed"))->value();
      if (random_seed < 0) {
        random_seed = (boost::posix_time::microsec_clock::universal_time() -
                       boost::posix_time::ptime(boost::posix_time::min_date_time))
                      .total_milliseconds();
      }
      
      typedef boost::ecuyer1988 rng_t; // (2**50 = 1T samples, 1000 chains)
      rng_t base_rng(random_seed);
      
      // Advance generator to avoid process conflicts
      static boost::uintmax_t DISCARD_STRIDE = static_cast<boost::uintmax_t>(1) << 50;
      base_rng.discard(DISCARD_STRIDE * (id - 1));
      
      //////////////////////////////////////////////////
      //                  Input/Output                //
      //////////////////////////////////////////////////
      
      // Data input
      std::string data_file = dynamic_cast<string_argument*>(parser.arg("data"))->value();
      
      std::fstream data_stream(data_file.c_str(),
                               std::fstream::in);
      stan::io::dump data_var_context(data_stream);
      data_stream.close();
      
      // Sample output
      std::string sample_file = dynamic_cast<string_argument*>(
                                parser.arg("output")->arg("sample"))->value();

      
      bool append_samples = dynamic_cast<bool_argument*>(
                            parser.arg("output")->arg("append_sample"))->value();
      
      std::ios_base::openmode samples_append_mode
        = append_samples
          ? (std::fstream::out | std::fstream::app)
          : std::fstream::out;
      
      std::fstream* sample_stream = 0;
      if(sample_file != "") {
        std::fstream sample(sample_file.c_str(),
                            samples_append_mode);
        sample_stream = &sample;
      }
      
      // Diagnostic output
      std::string diagnostic_file = dynamic_cast<string_argument*>(
                                    parser.arg("output")->arg("diagnostic"))->value();
      
      bool append_diagnostic = dynamic_cast<bool_argument*>(
                               parser.arg("output")->arg("append_diagnostic"))->value();
      
      std::ios_base::openmode diagnostic_append_mode
        = append_diagnostic
          ? (std::fstream::out | std::fstream::app)
          : std::fstream::out;
      
      std::fstream* diagnostic_stream = 0;
      if(diagnostic_file != "") {
        std::fstream diagnostic(diagnostic_file.c_str(),
                                diagnostic_append_mode);
        diagnostic_stream = &diagnostic;
      }
      
      parser.print(&std::cout);
      
      return 0;
 
    }

  } // namespace prob


} // namespace stan

#endif
