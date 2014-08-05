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
       * Writes out a vector as string.
       */
      class resume {
      private:
        std::ostream *o_;
        const bool has_stream_;
        const std::string prefix_;
      
      public:
        /**
         * Construct an object.
         *
         * @param o pointer to stream. Will accept 0.
         */
        resume(std::ostream *o, std::string prefix) 
          : o_(o), has_stream_(o != 0), prefix_(prefix) { }
        
        template <class RNG>
        void save_rng(const RNG& base_rng) {
          if (!has_stream_)
            return;
          
          //save rng state
          *o_ << prefix_ << "rng" << std::endl;
          *o_ << base_rng;
          *o_ << std::endl << std::endl;  
        }        

      
        template <class Model, class RNG>
        void save_inits(const Model& model, const RNG& base_rng, stan::mcmc::sample s) {
          if (!has_stream_)
            return;

          *o_ << prefix_ << "inits" << std::endl;
          
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
               *o_ << ", ";            
            else {
              
              *o_ << cur_param_name;
              *o_ << " <- ";
              
              if (cur_param_name != pnames.at(i)) //then dim > 1
                *o_ << "structure(c(";          

            }
            
              
            *o_ << std::fixed << std::setprecision(std::numeric_limits<double>::digits10) << model_values(i);
              
            
            prev_param_name = cur_param_name;
            if (i+1 < mv_size) {
              next_param_name = boost::regex_replace(pnames.at(i+1), regexp1, "");
              if(cur_param_name == next_param_name)
                continue;
            }
            
            if (cur_param_name == pnames.at(i)) //then dim == 1
              *o_ << std::endl;
            else {
              *o_ << "), .Dim = c(";

              *o_ <<
                boost::regex_replace(boost::regex_replace(
                pnames.at(i), regexp2, ""), regexp3, ",");
                  
              *o_ << "))" << std::endl;
            }
            
          } //for loop end

          *o_ << std::endl;
          //save inits end
        }
        
        void save_sampler_specific(stan::mcmc::base_mcmc* sampler) {
          if (!has_stream_)
            return;
          std::stringstream stream;
          sampler->write_sampler_specific_resume_info(&stream);
          *o_ << stream.str() << std::endl;
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
          if (!has_stream_)
            return;
          *o_ << prefix_ << x << std::endl;
        }
      
        /**
         * Prints a blank line. No prefix, no nothing.
         *
         */
        void operator()() {
          if (!has_stream_)
            return;
          *o_ << std::endl;
        }
      
        /**
         * Indicator function for whether the instance is recording.
         *
         * For this class, returns true if it has a stream.
         */
        bool is_recording() const {
          return has_stream_;
        }
      };


    }
  }
}

#endif
