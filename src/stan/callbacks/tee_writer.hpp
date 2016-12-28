#ifndef STAN_CALLBACKS_TEE_WRITER_HPP
#define STAN_CALLBACKS_TEE_WRITER_HPP

#include <stan/callbacks/writer.hpp>
#include <ostream>
#include <vector>
#include <string>

namespace stan {
  namespace callbacks {

    /**
     * <code>tee_writer</code> is an implementation that writes to
     * multiple writers.
     *
     * For any given call to this writer, it will delegate to both writers
     * provided in the constructor.
     */
    class tee_writer : public writer {
    public:
      /**
       * Constructor.
       *
       * @param[in, out] writer1 first writer
       * @param[in, out] writer2 second writer
       */
      tee_writer(writer& writer1,
                 writer& writer2)
        : writer1_(writer1), writer2_(writer2) {
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
      writer& writer1_;
      writer& writer2_;
    };

  }
}
#endif
