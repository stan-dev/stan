#ifndef STAN__COMMON__WRITE_MODEL_HPP
#define STAN__COMMON__WRITE_MODEL_HPP

#include <ostream>
#include <string>

namespace stan {

  namespace common {

    void write_model(std::ostream* s, 
                     const std::string model_name, 
                     const std::string prefix = "") {
      if (!s) return;
      
      *s << prefix << " model = " << model_name << std::endl;
    }
    

  } // namespace ui

} // namespace stan

#endif
