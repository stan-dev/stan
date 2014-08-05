#ifndef STAN__COMMON__RECORDER__RESUME_HPP
#define STAN__COMMON__RECORDER__RESUME_HPP

#include <ostream>
#include <string>
#include <vector>
#include <boost/regex.hpp>

namespace stan {
  namespace common {
    namespace recorder {
      
      /**
       * Records and loads sampler resume information.
       */
      class resume {
      private:
        std::fstream *save_;
        std::fstream *load_;
        const bool has_save_stream_;
        const bool has_load_stream_;
        const std::string prefix_;
      
      public:
        /**
         * Construct an object.
         *
         * @param o pointer to stream. Will accept 0.
         */
        resume(std::fstream *save_s, std::fstream *load_s, std::string prefix) 
          : save_(save_s), load_(load_s),
            has_save_stream_(save_s != 0), 
            has_load_stream_(load_s != 0),
            prefix_(prefix) { }
        
        template <class RNG>
        void save_rng(const RNG& base_rng) {
          if (!has_save_stream_)
            return;
          
          //save rng state
          *save_ << prefix_ << "rng" << std::endl;
          *save_ << base_rng;
          *save_ << std::endl << prefix_ << "end" <<  std::endl;  
        }      
          
        template <class RNG>
        void load_rng(RNG& base_rng) {
          if (!has_load_stream_)
            return;
          
          //load
          std::stringstream rng_stream;
              
          bool started = false;
          std::string line;
          while (std::getline(*load_, line)) {
            if (started) {
              if (line == "//end")
                break;                
              rng_stream << line;
            }
            if (line == "//rng")
              started = true;
          }
                  
          rng_stream >> base_rng;
        }        

      
        template <class Model, class RNG>
        void save_inits(const Model& model, const RNG& base_rng, stan::mcmc::sample s) {
          if (!has_save_stream_)
            return;

          *save_ << prefix_ << "inits" << std::endl;
          
          std::vector<std::string> pnames;
          model.constrained_param_names(pnames, false, false); 
                                    
          Eigen::VectorXd model_values;
          model.write_array(base_rng,
                            const_cast<Eigen::VectorXd&>(s.cont_params()),
                            model_values,
                            false, false);
                            
          //for (size_t i = 0; i < pnames.size(); i++)
          //  std::cout << pnames[i] << std::endl;          
          //for (size_t i = 0; i < model_values.size(); i++)
          //  std::cout << model_values(i) << std::endl;
          
          std::string prev_param_name;
          std::string cur_param_name;
          std::string next_param_name;
          boost::regex regexp1("\\..*");
          boost::regex regexp2("^.*?\\.");
          boost::regex regexp3("\\.");
          size_t mv_size = model_values.size();
          
          if (mv_size > 0)
            next_param_name = boost::regex_replace(pnames.at(0), regexp1, "");
          for (size_t i = 0; i < mv_size; i++) {
            cur_param_name = next_param_name;
            
            //check if we have a new param name
            if (prev_param_name == cur_param_name)
               *save_ << ", ";            
            else {
              
              *save_ << cur_param_name;
              *save_ << " <- ";
              
              if (cur_param_name != pnames.at(i)) //then dim > 1
                *save_ << "structure(c(";          

            }
            
              
            *save_ << std::fixed << std::setprecision(std::numeric_limits<double>::digits10) << model_values(i);
              
            
            prev_param_name = cur_param_name;
            if (i+1 < mv_size) {
              next_param_name = boost::regex_replace(pnames.at(i+1), regexp1, "");
              if(cur_param_name == next_param_name)
                continue;
            }
            
            if (cur_param_name == pnames.at(i)) //then dim == 1
              *save_ << std::endl;
            else {
              *save_ << "), .Dim = c(";

              *save_ <<
                boost::regex_replace(boost::regex_replace(
                pnames.at(i), regexp2, ""), regexp3, ",");
                  
              *save_ << "))" << std::endl;
            }
            
          } //for loop end

          *save_ << prefix_ << "end" <<  std::endl;
          //save inits end
        }
        
        void save_sampler_specific(stan::mcmc::base_mcmc* sampler) {
          if (!has_save_stream_)
            return;
          std::stringstream stream;
          sampler->write_sampler_specific_resume_info(&stream);
          *save_ << stream.str() << prefix_ << "end" <<  std::endl;
        }
        
        void load_sampler_specific(stan::mcmc::base_mcmc* sampler) {
          if (!has_load_stream_)
            return;
          sampler->load_sampler_specific_resume_info(load_);
        }

      
        /**
         * Print single string with a prefix
         *
         * Uses the insertion operator to write out a string
         * as comma separated values, flushing the buffer after the
         * line is complete
         * 
         * @param x string to print with prefix in front
         */
        void operator()(const std::string x) {
          if (!has_save_stream_)
            return;
          *save_ << prefix_ << x << std::endl;
        }
      
        /**
         * Prints a blank line. No prefix, no nothing.
         *
         */
        void operator()() {
          if (!has_save_stream_)
            return;
          *save_ << std::endl;
        }
      
        bool is_recording_save() const {
          return has_save_stream_;
        }
        
        bool is_recording_load() const {
          return has_load_stream_;
        }
      };


    }
  }
}

#endif
