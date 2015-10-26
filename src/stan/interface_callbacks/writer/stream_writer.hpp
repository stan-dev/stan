#ifndef STAN_INTERFACE_CALLBACKS_WRITER_STREAM_WRITER_HPP
#define STAN_INTERFACE_CALLBACKS_WRITER_STREAM_WRITER_HPP

#include <stan/interface_callbacks/writer/base_writer.hpp>
#include <ostream>
#include <vector>
#include <string>

namespace stan {
  namespace interface_callbacks {
    namespace writer {

      /**
       * stream_writer writes to an std::ostream.
       */
      class stream_writer : public base_writer {
      public:
        /**
         * Constructor.
         *
         * @param output std::ostream to write to
         * @param key_value_prefix String to write before lines
         *   treated as comments.
         */
        stream_writer(std::ostream& output,
                      const std::string& key_value_prefix = ""):
          output__(output), key_value_prefix__(key_value_prefix) {}

        void operator()(const std::string& key, double value) {
          output__ << key_value_prefix__ << key << " = " << value << std::endl;
        }

        void operator()(const std::string& key, int value) {
          output__ << key_value_prefix__ << key << " = " << value << std::endl;
        }

        void operator()(const std::string& key, const std::string& value) {
          output__ << key_value_prefix__ << key << " = " << value << std::endl;
        }

        void operator()(const std::string& key,
                        const double* values,
                        int n_values) {
          if (n_values == 0) return;

          output__ << key_value_prefix__ << key << ": ";

          output__ << values[0];
          for (int n = 1; n < n_values; ++n)
            output__ << "," << values[n];
          output__ << std::endl;
        }

        void operator()(const std::string& key,
                        const double* values,
                        int n_rows, int n_cols) {
          if (n_rows == 0 || n_cols == 0) return;

          output__ << key_value_prefix__ << key << std::endl;

          for (int i = 0; i < n_rows; ++i) {
            output__ << key_value_prefix__ << values[i * n_cols];
            for (int j = 1; j < n_cols; ++j)
              output__ << "," << values[i * n_cols + j];
            output__ << std::endl;
          }
        }

        void operator()(const std::vector<std::string>& names) {
          if (names.empty()) return;

          std::vector<std::string>::const_iterator last = names.end();
          --last;

          for (std::vector<std::string>::const_iterator it = names.begin();
               it != last; ++it)
            output__ << *it << ",";
          output__ << names.back() << std::endl;
        }

        void operator()(const std::vector<double>& state) {
          if (state.empty()) return;

          std::vector<double>::const_iterator last = state.end();
          --last;

          for (std::vector<double>::const_iterator it = state.begin();
               it != last; ++it)
            output__ << *it << ",";
          output__ << state.back() << std::endl;
        }

        void operator()() {
          output__ << key_value_prefix__ << std::endl;
        }

        void operator()(const std::string& message) {
          output__ << key_value_prefix__ << message << std::endl;
        }

      private:
        std::ostream& output__;
        std::string key_value_prefix__;
      };

    }
  }
}

#endif
