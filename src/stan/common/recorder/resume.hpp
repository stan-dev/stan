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
        std::ostream *save_;
        std::istream *load_;
        const bool has_save_stream_;
        const bool has_load_stream_;
        const std::string prefix_;
        const std::string end_;
      
      public:
        /**
         * Construct an object.
         *
         * @param o pointer to stream. Will accept 0.
         */
        resume(std::ostream *save_s, std::istream *load_s, std::string prefix) 
          : save_(save_s), load_(load_s),
            has_save_stream_(save_s != 0), 
            has_load_stream_(load_s != 0),
            prefix_(prefix),
            end_(prefix+"end") { }
        
        void save_common(const std::string& name, const std::iostream& input_stream) {
          if (!has_save_stream_)
            return;            

          *save_ << prefix_ << name << std::endl;
          *save_ << input_stream.rdbuf();
          *save_ << std::endl << end_ <<  std::endl;  
        }
        
        void save_double(const std::string& name, const double input_double) {
          if (!has_save_stream_)
            return;          
          
          std::stringstream double_stream;
          
          double_stream << std::fixed << std::setprecision(std::numeric_limits<double>::digits10) << input_double;
          
          save_common(name, double_stream); 
        }

        void load_common(const std::string& name, std::iostream& output_stream) {
          if (!has_load_stream_)
            return;            
              
          bool started = false;
          std::string line;
          std::string begin_ = prefix_+name;
          while (std::getline(*load_, line)) {
            if (started) {
              if (line == end_)
                break;                
              output_stream << line;
            }
            if (line == begin_)
              started = true;
          }
        }
        
        void load_double(const std::string& name, double& output_double) {
          std::stringstream double_stream;
          load_common(name, double_stream);
          double_stream >> output_double;
        }

        template <class RNG>
        void save_rng(const RNG& base_rng) {
          if (!has_save_stream_)
            return;
          
          std::stringstream rng_stream;
          rng_stream << base_rng;
          save_common("rng", rng_stream);  
        }
        
        template <class RNG>
        void load_rng(RNG& base_rng) {
          if (!has_load_stream_)
            return;
          
          std::stringstream rng_stream;
          load_common("rng", rng_stream);
          rng_stream >> base_rng;
        }

      
        template <class Model, class RNG>
        void save_inits(const Model& model, const RNG& base_rng, stan::mcmc::sample s) {
          if (!has_save_stream_)
            return;
          
          std::stringstream inits_stream;
          
          std::vector<std::string> pnames;
          model.constrained_param_names(pnames, false, false); 
                                    
          Eigen::VectorXd model_values;
          model.write_array(base_rng,
                            const_cast<Eigen::VectorXd&>(s.cont_params()),
                            model_values,
                            false, false);
          
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
               inits_stream << ", ";            
            else {
              
              inits_stream << cur_param_name;
              inits_stream << " <- ";
              
              if (cur_param_name != pnames.at(i)) //then dim > 1
                inits_stream << "structure(c(";          

            }
            
              
            inits_stream << std::fixed << std::setprecision(std::numeric_limits<double>::digits10) << model_values(i);
              
            
            prev_param_name = cur_param_name;
            if (i+1 < mv_size) {
              next_param_name = boost::regex_replace(pnames.at(i+1), regexp1, "");
              if(cur_param_name == next_param_name)
                continue;
            }
            
            if (cur_param_name == pnames.at(i)) //then dim == 1
              inits_stream << std::endl;
            else {
              inits_stream << "), .Dim = c(";

              inits_stream <<
                boost::regex_replace(boost::regex_replace(
                pnames.at(i), regexp2, ""), regexp3, ",");
                  
              inits_stream << "))" << std::endl;
            }
            
          } //for loop end

          save_common("inits", inits_stream);
        }
        
        template <class Sampler>
        void save_sampler_specific(Sampler* sampler) {
          if (!has_save_stream_)
            return;
          sampler->save_sampler_specific_resume_info(this);
        }
        
        template <class Sampler>        
        void load_sampler_specific(Sampler* sampler) {
          if (!has_load_stream_)
            return;
          sampler->load_sampler_specific_resume_info(this);
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
