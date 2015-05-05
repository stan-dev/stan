#ifndef STAN__IO__BASE_TRANSFORM_HPP
#define STAN__IO__BASE_TRANSFORM_HPP

#include <string>
#include <vector>

namespace stan {
  namespace io {

    class base_transform {
      static void constrain(const vector<double>& input,
                            vector<double>& output) {}
      static void unconstrain(const vector<double>& input,
                              vector<double>& output) {}
      static int constrained_dim() { return 0; }
      static int unconstrained_dim() { return 0; }
      static std::string base_type() { return "null"; }
      // turn base type into an enum
    }

  }  // io
}  // stan

#endif
