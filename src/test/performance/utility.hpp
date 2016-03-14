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

#include <stan/mcmc/hmc/static/adapt_unit_e_static_hmc.hpp>
#include <stan/mcmc/hmc/static/adapt_diag_e_static_hmc.hpp>
#include <stan/mcmc/hmc/static/adapt_dense_e_static_hmc.hpp>
#include <stan/mcmc/hmc/nuts/adapt_unit_e_nuts.hpp>
#include <stan/mcmc/hmc/nuts/adapt_diag_e_nuts.hpp>
#include <stan/mcmc/hmc/nuts/adapt_dense_e_nuts.hpp>

#include <stan/model/util.hpp>

#include <stan/optimization/newton.hpp>
#include <stan/optimization/bfgs.hpp>

#include <stan/services/init/initialize_state.hpp>
#include <stan/services/io/do_print.hpp>
#include <stan/services/io/write_error_msg.hpp>
#include <stan/services/io/write_iteration.hpp>
#include <stan/services/io/write_model.hpp>
#include <stan/services/io/write_stan.hpp>
#include <stan/services/sample/generate_transitions.hpp>
#include <stan/services/sample/progress.hpp>
#include <stan/services/sample/hmc_nuts_diag_e_adapt.hpp>

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
        interface_callbacks::interrupt::noop interrupt;
        
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


        // Sampling parameters
        int num_thin = 1;
        bool save_warmup = false;
        double stepsize = 1.0;
        double stepsize_jitter = 0.0;
        int max_depth = 10;
        double delta = 0.8;
        double gamma = 0.05;
        double kappa = 0.75;
        double t0 = 10;
        unsigned int init_buffer = 75;
        unsigned int term_buffer = 50;
        unsigned int window = 25;
        stan::services::sample::hmc_nuts_diag_e_adapt(model,
                                                      base_rng,
                                                      cont_params,
                                                      num_warmup,
                                                      num_samples,
                                                      num_thin,
                                                      save_warmup,
                                                      refresh,
                                                      stepsize,
                                                      stepsize_jitter,
                                                      max_depth,
                                                      delta,
                                                      gamma,
                                                      kappa,
                                                      t0,
                                                      init_buffer,
                                                      term_buffer,
                                                      window,
                                                      interrupt,
                                                      sample_writer,
                                                      diagnostic_writer,
                                                      info);

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
