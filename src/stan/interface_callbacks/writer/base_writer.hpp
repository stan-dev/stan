#ifndef STAN_INTERFACE_CALLBACKS_WRITER_BASE_WRITER_HPP
#define STAN_INTERFACE_CALLBACKS_WRITER_BASE_WRITER_HPP

#include <string>
#include <vector>

namespace stan {
  namespace interface_callbacks {
    namespace writer {

      /**
       * base_writer
       *
       * Abstract class for writing different things to stream.
       */
      class base_writer {
      public:

        /**
         * Destructor.
         */
        virtual ~base_writer() {}

        /**
         * Writes a key value pair.
         *
         * @param[in] key The key.
         * @param[in] value The value to write.
         */
        virtual void operator()(const std::string& key,
                                double value) { }

        /**
         * Writes a key value pair.
         *
         * @param[in] key The key.
         * @param[in] value The value to write.
         */
        virtual void operator()(const std::string& key,
                                const std::string& value) { }

        /**
         * Writes the key and an array of values associated with the key.
         *
         * @param[in] key The key.
         * @param[in] values an array of values.
         * @param[in] n_values the number of values.
         */
        virtual void operator()(const std::string& key,
                                const double* values,
                                int n_values) { }


        /**
         * Writes the key and a 2-d array of values associated with
         * the key.
         *
         * @param[in] key The key.
         * @param[in] values The 2-d array of values in row-major order
         * @param[in] n_rows The number of rows
         * @param[in] n_cols The number of columns.
         */
        virtual void operator()(const std::string& key,
                                const double* values,
                                int n_rows, int n_cols) { }

        /**
         * Writes names.
         *
         * @param[in] names The names as a vector.
         */
        virtual void operator()(const std::vector<std::string>& names) { }

        /**
         * Writes the state.
         *
         * @param[in] state The state as a vector.
         */
        virtual void operator()(const std::vector<double>& state) { }

        /**
         * Writes nothing.
         */
        virtual void operator()() { }

        /**
         * Writes a message.
         *
         * @param[in] message Message
         */
        virtual void operator()(const std::string& message) { }

        /**
         * Reports whether the writer is active.
         *
         * @return true if the writer is active, false otherwise
         */
        virtual bool is_writing() const = 0;
      };

    }
  }
}

#endif
