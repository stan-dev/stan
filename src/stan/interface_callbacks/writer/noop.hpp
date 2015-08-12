#ifndef STAN_INTERFACE_CALLBACKS_WRITER_NOOP_HPP
#define STAN_INTERFACE_CALLBACKS_WRITER_NOOP_HPP

#include <stan/interface_callbacks/writer/base_writer.hpp>
#include <string>
#include <vector>

namespace stan {
  namespace interface_callbacks {
    namespace writer {

      /**
       * No op writer.
       *
       */
      class noop: public base_writer {
      public:
        /**
         * Writes a key, value pair.
         *
         * @param key The key
         * @param value The value
         */
        void operator()(const std::string& key,
                        double value) { }

        /**
         * Writes a key, value pair.
         *
         * @param key The key
         * @param value The value
         */
        void operator()(const std::string& key,
                        const std::string& value) { }

        /**
         * Writes a key and an array of values.
         *
         * @param key The key
         * @param values The values
         * @param n_values The number of values
         */
        void operator()(const std::string& key,
                        const double* values,
                        int n_values)  { }

        /**
         * Writes a key and a 2-d array of values.
         *
         * @param key The key
         * @param values The values stored in row-major order
         * @param n_rows The number of rows
         * @param n_cols The number of columns
         */
        void operator()(const std::string& key,
                        const double* values,
                        int n_rows, int n_cols) { }

        /**
         * Writes a vector of names.
         *
         * @param names Names to write
         */
        void operator()(const std::vector<std::string>& names) { }

        /**
         * Writes a vector of states.
         *
         * @param state State to write
         */
        void operator()(const std::vector<double>& state) { }

        /**
         * Writes a message.
         *
         * @param message Message to write
         */
        void operator()(const std::string& message) { }

        /**
         * Writes nothing.
         *
         * An implementation may choose to treat this as a flush.
         */
        void operator()() { }
      };

    }
  }
}

#endif
