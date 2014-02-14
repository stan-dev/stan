#ifndef __STAN__UI__WRITE_MODEL_HPP__
#define __STAN__UI__WRITE_MODEL_HPP__

#include <ostream>
#include <string>

namespace stan {

  namespace ui {

    void write_model(std::ostream* s, 
                     const std::string model_name, 
                     const std::string prefix = "") {
      if (!s) return;
      
      *s << prefix << " model = " << model_name << std::endl;
    }
    

  } // namespace ui

} // namespace stan

#endif
