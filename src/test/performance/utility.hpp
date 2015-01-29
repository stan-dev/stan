#ifndef TEST__PERFORMANCE__UTILITY_HPP
#define TEST__PERFORMANCE__UTILITY_HPP

#include <stdexcept>
#include <boost/algorithm/string.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>

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
        // captures both std::cout amd std::err
  
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
        for (int n = 0; n < draw.size(); ++n) {
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


    }
  }
}
#endif
