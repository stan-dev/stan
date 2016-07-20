#ifndef STAN_CALLBACKS_WRITER_CHAINED_WRITER_HPP
#define STAN_CALLBACKS_WRITER_CHAINED_WRITER_HPP

#include <stan/callbacks/writer/base_writer.hpp>
#include <ostream>
#include <vector>
#include <string>

namespace stan {
  namespace callbacks {
    namespace writer {

      /**
       * stream_writer writes to an std::ostream.
       */
      class chained_writer : public base_writer {
      public:
        /**
         * Constructor.
         *
         * @param writer1 first writer
         * @param writer2 second writer
         */
        chained_writer(base_writer& writer1,
                       base_writer& writer2)
          : writer1_(writer1), writer2_(writer2) {
        }


        void operator()(const std::string& key, double value) {
          writer1_(key, value);
          writer2_(key, value);
        }

        void operator()(const std::string& key, int value) {
          writer1_(key, value);
          writer2_(key, value);
        }

        void operator()(const std::string& key, const std::string& value) {
          writer1_(key, value);
          writer2_(key, value);
        }

        void operator()(const std::string& key,
                        const double* values,
                        int n_values) {
          writer1_(key, values, n_values);
          writer2_(key, values, n_values);
        }

        void operator()(const std::string& key,
                        const double* values,
                        int n_rows, int n_cols) {
          writer1_(key, values, n_rows, n_cols);
          writer2_(key, values, n_rows, n_cols);
        }

        void operator()(const std::vector<std::string>& names) {
          writer1_(names);
          writer2_(names);
        }

        void operator()(const std::vector<double>& state) {
          writer1_(state);
          writer2_(state);
        }

        void operator()() {
          writer1_();
          writer2_();
        }

        void operator()(const std::string& message) {
          writer1_(message);
          writer2_(message);
        }

      private:
        base_writer& writer1_;
        base_writer& writer2_;
      };

    }
  }
}

#endif
