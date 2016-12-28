#ifndef STAN_CALLBACKS_STREAM_WRITER_HPP
#define STAN_CALLBACKS_STREAM_WRITER_HPP

#include <stan/callbacks/writer.hpp>
#include <ostream>
#include <vector>
#include <string>

namespace stan {
  namespace callbacks {

    /**
     * <code>stream_writer</code> writes to an <code>std::ostream</code>.
     */
    class stream_writer : public writer {
    public:
      /**
       * Constructor.
       *
       * @param[in, out] output std::ostream to write
       * @param[in] key_value_prefix String to write before lines
       *   treated as comments.
       */
      stream_writer(std::ostream& output,
                    const std::string& key_value_prefix = ""):
        output_(output), key_value_prefix_(key_value_prefix) {}

      void operator()(const std::vector<std::string>& names) {
        if (names.empty()) return;

        std::vector<std::string>::const_iterator last = names.end();
        --last;

        for (std::vector<std::string>::const_iterator it = names.begin();
             it != last; ++it)
          output_ << *it << ",";
        output_ << names.back() << std::endl;
      }

      void operator()(const std::vector<double>& state) {
        if (state.empty()) return;

        std::vector<double>::const_iterator last = state.end();
        --last;

        for (std::vector<double>::const_iterator it = state.begin();
             it != last; ++it)
          output_ << *it << ",";
        output_ << state.back() << std::endl;
      }

      void operator()() {
        output_ << key_value_prefix_ << std::endl;
      }

      void operator()(const std::string& message) {
        output_ << key_value_prefix_ << message << std::endl;
      }

    private:
      std::ostream& output_;
      std::string key_value_prefix_;
    };

  }
}
#endif
