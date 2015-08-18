#ifndef STAN_INTERFACE_CALLBACKS_WRITER_MESSAGES_HPP
#define STAN_INTERFACE_CALLBACKS_WRITER_MESSAGES_HPP

#include <stan/interface_callbacks/writer/base_writer.hpp>
#include <ostream>
#include <string>
#include <vector>

namespace stan {
  namespace interface_callbacks {
    namespace writer {

      /**
       * Writes messages.
       * 
       * This only outputs messages. It does not handle any of
       * the key, value pairs or the states.
       */
      class messages : public base_writer {
      private:
        std::ostream *o_;
        const bool has_stream_;
        const std::string prefix_;

      public:
        /**
         * Constructs this writer with a output stream and a prefix.
         *
         * @param o a pointer to an output stream. This can be NULL.
         * @param prefix a string that will be printed before each line.
         */
        messages(std::ostream *o, const std::string& prefix)
          : o_(o), has_stream_(o != 0), prefix_(prefix) { }

        /**
         * Constructs this writer with an output stream.
         *
         * @param o a pointer to an output stream. This can be NULL.
         */
        explicit messages(std::ostream *o)
          : o_(o), has_stream_(o != 0), prefix_("") { }

        /**
         * Writes a key, value pair.
         * This implementation does nothing.
         *
         * @param key The key
         * @param value The value
         */
        void operator()(const std::string& key,
                        double value) { }

        /**
         * Writes a key, value pair.
         * This implementation does nothing.
         *
         * @param key The key
         * @param value The value
         */
        void operator()(const std::string& key,
                        const std::string& value) { }

        /**
         * Writes a key and an array of values.
         * This implementation does nothing.
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
         * This implementation does nothing.
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
         * This implementation does nothing.
         *
         * @param names Names to write
         */
        void operator()(const std::vector<std::string>& names) { }

        /**
         * Writes a vector of states.
         * This implementation does nothing.
         *
         * @param state State to write
         */
        void operator()(const std::vector<double>& state) { }

        /**
         * Writes a message.
         *
         * @param message Message to write
         */
        void operator()(const std::string& message) {
          if (has_stream_)
            *o_ << prefix_ << message << std::endl;
        }
        
        /**
         * Writes nothing.
         */
        void operator()() {
          if (has_stream_)
            *o_ << prefix_ << std::endl;
        }
      };
    }
  }
}

#endif
