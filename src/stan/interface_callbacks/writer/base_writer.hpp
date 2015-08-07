#ifndef STAN_INTERFACE_CALLBACKS_WRITER_BASE_WRITER_HPP
#define STAN_INTERFACE_CALLBACKS_WRITER_BASE_WRITER_HPP

#include <string>
#include <vector>

namespace stan {
  namespace interface_callbacks {
    namespace writer {

      class base_writer {
      public:
        virtual ~base_writer() {}

        virtual void operator()(const std::string& key,
                                double value) { }
        virtual void operator()(const std::string& key,
                                const std::string& value) { }
        virtual void operator()(const std::string& key,
                                const double* values,
                                int n_values) { }
        virtual void operator()(const std::string& key,
                                const double* values,
                                int n_rows, int n_cols) { }
        
        virtual void operator()(const std::vector<std::string>& names) { }
        virtual void operator()(const std::vector<double>& state) { }
        
        virtual void operator()() { }
        virtual void operator()(const std::string& message) { }
        
        virtual bool is_writing() const = 0;
      };

    }
  }
}

#endif
