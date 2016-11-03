#ifndef TEST__PERFORMANCE__UTILITY_HPP
#define TEST__PERFORMANCE__UTILITY_HPP

#include <stdexcept>
#include <boost/algorithm/string.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <boost/random/additive_combine.hpp>  // L'Ecuyer RNG
#include <boost/random/uniform_real_distribution.hpp>

#include <stan/version.hpp>
#include <stan/io/cmd_line.hpp>
#include <stan/io/dump.hpp>
#include <stan/io/json/json_data.hpp>
#include <stan/io/json/json_data_handler.hpp>
#include <stan/io/json/json_error.hpp>
#include <stan/io/json/json_handler.hpp>
#include <stan/io/json/json_parser.hpp>
#include <stan/services/sample/mcmc_writer.hpp>

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
#include <stan/mcmc/fixed_param_sampler.hpp>
#include <stan/mcmc/hmc/static/adapt_unit_e_static_hmc.hpp>
#include <stan/mcmc/hmc/static/adapt_diag_e_static_hmc.hpp>
#include <stan/mcmc/hmc/static/adapt_dense_e_static_hmc.hpp>
#include <stan/mcmc/hmc/nuts/adapt_unit_e_nuts.hpp>
#include <stan/mcmc/hmc/nuts/adapt_diag_e_nuts.hpp>
#include <stan/mcmc/hmc/nuts/adapt_dense_e_nuts.hpp>

#include <stan/model/gradient.hpp>

#include <stan/optimization/newton.hpp>
#include <stan/optimization/bfgs.hpp>

#include <stan/services/init/initialize_state.hpp>
#include <stan/services/io/do_print.hpp>
#include <stan/services/io/write_error_msg.hpp>
#include <stan/services/io/write_iteration.hpp>
#include <stan/services/io/write_model.hpp>
#include <stan/services/io/write_stan.hpp>
#include <stan/services/mcmc/sample.hpp>
#include <stan/services/mcmc/warmup.hpp>
#include <stan/services/optimize/do_bfgs_optimize.hpp>
#include <stan/services/sample/generate_transitions.hpp>
#include <stan/services/sample/init_adapt.hpp>
#include <stan/services/sample/init_nuts.hpp>
#include <stan/services/sample/init_static_hmc.hpp>
#include <stan/services/sample/init_windowed_adapt.hpp>
#include <stan/services/sample/progress.hpp>

// FIXME: These belong to the interfaces and should be templated out here
#include <stan/interface_callbacks/interrupt/noop.hpp>
#include <stan/interface_callbacks/var_context_factory/dump_factory.hpp>
#include <stan/interface_callbacks/writer/noop_writer.hpp>
#include <stan/interface_callbacks/writer/stream_writer.hpp>
#include <stan/interface_callbacks/writer/base_writer.hpp>

#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace stan {
  namespace test {
    namespace performance {

      struct run_command_output {
        std::string command;
        std::string output;
        long time;
        int err_code;
        bool hasError;
        std::string header;
        std::string body;

        run_command_output(const std::string command,
                           const std::string output,
                           const long time,
                           const int err_code)
          : command(command),
            output(output),
            time(time),
            err_code(err_code),
            hasError(err_code != 0),
            header(),
            body()
        { 
          size_t end_of_header = output.find("\n\n");
          if (end_of_header == std::string::npos)
            end_of_header = 0;
          else
            end_of_header += 2;
          header = output.substr(0, end_of_header);
          body = output.substr(end_of_header);
        }
  
        run_command_output() 
          : command(),
            output(),
            time(0),
            err_code(0),
            hasError(false),
            header(),
            body()
        { }
      };

      std::ostream& operator<<(std::ostream& os, const run_command_output& out) {
        os << "run_command output:" << "\n"
           << "- command:   " << out.command << "\n"
           << "- output:    " << out.output << "\n"
           << "- time (ms): " << out.time << "\n"
           << "- err_code:  " << out.err_code << "\n"
           << "- hasError:  " << (out.hasError ? "true" : "false") << "\n"
           << "- header:    " << out.header << "\n"
           << "- body:      " << out.body << std::endl;
        return os;
      }

      /** 
       * Runs the command provided and returns the system output
       * as a string.
       * 
       * @param command A command that can be run from the shell
       * @return the system output of the command
       */  
      run_command_output run_command(std::string command) {
        using boost::posix_time::ptime;
        using boost::posix_time::microsec_clock;
  
        FILE *in;
        std::string new_command = command + " 2>&1"; 
        // captures both cout amd err
  
        in = popen(command.c_str(), "r");
  
        if(!in) {
          std::string err_msg;
          err_msg = "Fatal error with popen; could not execute: \"";
          err_msg+= command;
          err_msg+= "\"";
          throw std::runtime_error(err_msg.c_str());
        }
  
        std::string output;
        char buf[1024];
        size_t count;
        ptime time_start(microsec_clock::universal_time()); // start timer
        while ((count = fread(&buf, 1, 1024, in)) > 0)
          output += std::string(&buf[0], &buf[count]);
        ptime time_end(microsec_clock::universal_time());   // end timer

        // bits 15-8 is err code, bit 7 if core dump, bits 6-0 is signal number
        int err_code = pclose(in);
        // on Windows, err code is the return code.
        if (err_code != 0 && (err_code >> 8) > 0)
          err_code >>= 8;

        return run_command_output(command, output,
                                  (time_end - time_start).total_milliseconds(), 
                                  err_code);
      }


      std::vector<double> get_last_iteration_from_file(const char* filename) {
        std::vector<double> draw;
        const char comment = '#';
        
        std::ifstream file_stream(filename);
        std::string line;
        std::string last_values;
        while (std::getline(file_stream, line)) {
          if (line.length() > 0 && line[0] != comment)
            last_values = line;
        }
        
        std::stringstream values_stream(last_values);
        std::vector<std::string> values;
        std::string value;
        while (std::getline(values_stream, value, ','))
          values.push_back(value);
        
        draw.resize(values.size());
        for (size_t n = 0; n < draw.size(); ++n) {
          draw[n] = atof(values[n].c_str());
        }
        
        return draw;
      }


      template <typename T>
      std::string quote(const T& val) {
        std::stringstream quoted_val;
        quoted_val << "\""
                   << val
                   << "\"";
        return quoted_val.str();
      }

      std::string get_git_hash() {
        run_command_output git_hash = run_command("git rev-parse HEAD");
        if (git_hash.hasError)
          return "NA";
        boost::trim(git_hash.body);
        return git_hash.body;
      }

      std::string get_git_date() {
        run_command_output git_date_command 
          = run_command("git log --format=%ct -1");
        if (git_date_command.hasError)
          return "NA";
        boost::trim(git_date_command.body);
  
        long timestamp = atol(git_date_command.body.c_str());
        std::time_t git_date(timestamp);
  
        std::stringstream date_ss;
        date_ss << std::ctime(&git_date);
        
        std::string date;
        date = date_ss.str();
        
        boost::trim(date);
        return date;
      }

      std::string get_date() {
        std::time_t curr_date;
        time(&curr_date);
        
        std::stringstream date_ss;
        date_ss << std::ctime(&curr_date);
        
        std::string date;
        date = date_ss.str();
        
        boost::trim(date);
        return date;
      }


      template <class Model>
      int command(int num_warmup,
                  int num_samples,
                  const std::string data_file, const std::string output_file,
                  unsigned int random_seed) {
        //////////////////////////////////////////////////
        //            Random number generator           //
        //////////////////////////////////////////////////

        typedef boost::ecuyer1988 rng_t;  // (2**50 = 1T samples, 1000 chains)
        rng_t base_rng(random_seed);

        stan::interface_callbacks::writer::stream_writer info(std::cout);
        

        // Advance generator to avoid process conflicts
        static boost::uintmax_t DISCARD_STRIDE
          = static_cast<boost::uintmax_t>(1) << 50;
        base_rng.discard(-DISCARD_STRIDE);

        //////////////////////////////////////////////////
        //                  Input/Output                //
        //////////////////////////////////////////////////

        // Data input
        std::fstream data_stream(data_file.c_str(),
                                 std::fstream::in);
        stan::io::dump data_var_context(data_stream);
        data_stream.close();

        // Sample output
        std::fstream* output_stream = 0;
        if (output_file != "") {
          output_stream = new std::fstream(output_file.c_str(),
                                           std::fstream::out);
        }
        std::fstream* diagnostic_stream = 0;

        // Refresh rate
        int refresh = num_samples;

        //////////////////////////////////////////////////
        //                Initialize Model              //
        //////////////////////////////////////////////////

        Model model(data_var_context, &std::cout);

        Eigen::VectorXd cont_params = Eigen::VectorXd::Zero(model.num_params_r());

        //parser.print(&std::cout);
        //std::cout << std::endl;

        std::string init = "0";
        //dynamic_cast<stan::services::string_argument*>(
        //parser.arg("init"))->value();

        interface_callbacks::writer::stream_writer sample_writer(*output_stream, "# ");
        interface_callbacks::writer::noop_writer diagnostic_writer;
        interface_callbacks::writer::stream_writer info_writer(std::cout, "# ");
        interface_callbacks::writer::stream_writer err_writer(std::cerr);

        if (output_stream) {
          services::io::write_stan(sample_writer);
          services::io::write_model(sample_writer, model.model_name());
          //parser.print(output_stream, "#");
        }

        interface_callbacks::var_context_factory::dump_factory var_context_factory;
        if (!services::init::initialize_state<interface_callbacks::var_context_factory::dump_factory>
            (init, cont_params, model, base_rng, info,
             var_context_factory))
          return stan::services::error_codes::SOFTWARE;

        //////////////////////////////////////////////////
        //              Sampling Algorithms             //
        //////////////////////////////////////////////////

        // Check timing
        clock_t start_check = clock();

        double init_log_prob;
        Eigen::VectorXd init_grad = Eigen::VectorXd::Zero(model.num_params_r());

        stan::model::gradient(model, cont_params, init_log_prob,
                              init_grad, &std::cout);

        clock_t end_check = clock();
        double deltaT
          = static_cast<double>(end_check - start_check) / CLOCKS_PER_SEC;

        std::cout << std::endl;
        std::cout << "Gradient evaluation took " << deltaT
                  << " seconds" << std::endl;
        std::cout << "1000 transitions using 10 leapfrog steps "
                  << "per transition would take "
                  << 1e4 * deltaT << " seconds." << std::endl;
        std::cout << "Adjust your expectations accordingly!"
                  << std::endl << std::endl;
        std::cout << std::endl;

        stan::services::sample::mcmc_writer<Model,
                                            interface_callbacks::writer::stream_writer,
                                            interface_callbacks::writer::noop_writer,
                                            interface_callbacks::writer::stream_writer>
          writer(sample_writer, diagnostic_writer, info_writer);

        // Sampling parameters
        int num_thin = 1;
        bool save_warmup = false;

        stan::mcmc::sample s(cont_params, 0, 0);

        double warmDeltaT;
        double sampleDeltaT;

        // Sampler


        bool adapt_engaged = true;

        typedef stan::mcmc::adapt_diag_e_nuts<Model, rng_t> sampler;
        sampler* sampler_ptr = 0;
        sampler_ptr = new sampler(model, base_rng);
        sampler_ptr->set_nominal_stepsize(1.0);
        sampler_ptr->set_stepsize_jitter(0.0);
        sampler_ptr->set_max_depth(10);

        stan::services::sample::init_adapt(sampler_ptr, 0.8, 0.05, 0.75, 10,
                                           cont_params,
                                           info_writer, err_writer);
        sampler_ptr->set_window_params(num_warmup, 75, 50, 25, info_writer);
          
        // Headers
        writer.write_sample_names(s, sampler_ptr, model);
        writer.write_diagnostic_names(s, sampler_ptr, model);
          
        std::string prefix = "";
        std::string suffix = "\n";
        interface_callbacks::interrupt::noop startTransitionCallback;
          
        // Warm-Up
        clock_t start = clock();
          
        services::mcmc::warmup<Model, rng_t>(sampler_ptr, num_warmup, num_samples, num_thin,
                                             refresh, save_warmup,
                                             writer,
                                             s, model, base_rng,
                                             prefix, suffix, std::cout,
                                             startTransitionCallback,
                                             info_writer,
                                             err_writer);

        clock_t end = clock();
        warmDeltaT = static_cast<double>(end - start) / CLOCKS_PER_SEC;
          
        if (adapt_engaged) {
          sampler_ptr->disengage_adaptation();
          writer.write_adapt_finish(sampler_ptr);
        }
          
        // Sampling
        start = clock();
          
        services::mcmc::sample<Model, rng_t>
          (sampler_ptr, num_warmup, num_samples, num_thin,
           refresh, true,
           writer,
           s, model, base_rng,
           prefix, suffix, std::cout,
           startTransitionCallback,
           info_writer,
           err_writer);
          
        end = clock();
        sampleDeltaT = static_cast<double>(end - start) / CLOCKS_PER_SEC;
          
        writer.write_timing(warmDeltaT, sampleDeltaT);
          
        if (sampler_ptr)
          delete sampler_ptr;

        
        if (output_stream) {
          output_stream->close();
          delete output_stream;
        }

        if (diagnostic_stream) {
          diagnostic_stream->close();
          delete diagnostic_stream;
        }
        
        return 0;
      }

    }
  }
}
#endif
