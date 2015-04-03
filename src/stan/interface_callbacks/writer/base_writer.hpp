#ifndef STAN_INTERFACE_CALLBACKS_WRITER_BASE_WRITER_HPP
#define STAN_INTERFACE_CALLBACKS_WRITER_BASE_WRITER_HPP

#include <string>
#include <vector>
#include <boost/lexical_cast.hpp>

namespace stan {
  namespace interface_callbacks {
    namespace writer {
      
      class base_writer {
      public:
        virtual ~base_writer() {};
        
        virtual void operator()(const std::string& key,
                                double value) = 0;
        virtual void operator()(const std::string& key,
                                const std::string& value) = 0;
        virtual void operator()(const std::string& key,
                                const double* values,
                                int n_values) = 0;
        virtual void operator()(const std::string& key,
                                const double* values,
                                int n_rows, int n_cols) = 0;
        
        virtual void operator()(const std::vector<std::string>& names) = 0;
        virtual void operator()(const std::vector<double>& state) = 0;
        
        virtual void operator()() = 0;
        virtual void operator()(const std::string& message) = 0;
        
      };

    }
  }
}

#endif
