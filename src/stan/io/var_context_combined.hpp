
#ifndef STAN__IO__VAR_CONTEXT_COMBINED_HPP
#define STAN__IO__VAR_CONTEXT_COMBINED_HPP

#include <stan/io/var_context.hpp>

namespace stan { 
  namespace io { 
    /**
     * A var_context_combined object represents two objects of var_context
     * as one. 
     */ 
    class var_context_combined : public  var_context {
    private:
      const var_context& vc1;
      const var_context& vc2;
    public:
      var_context_combined(const var_context& v1, const var_context& v2): vc1(v1), vc2(v2) {}
      virtual bool contains_i(const std::string& name) const {
        return vc1.contains_i(name) || vc2.contains_i(name);
      }
      virtual bool contains_r(const std::string& name) const {
        return vc1.contains_r(name) || vc2.contains_r(name);
      }
      virtual std::vector<double> vals_r(const std::string& name) const {
        return vc1.contains_r(name) ? vc1.vals_r(name) : vc2.vals_r(name);
      }
      virtual std::vector<int> vals_i(const std::string& name) const {
        return vc1.contains_i(name) ? vc1.vals_i(name) : vc2.vals_i(name);
      }
      virtual std::vector<size_t> dims_r(const std::string& name) const {
        return vc1.contains_r(name) ? vc1.dims_r(name) : vc2.dims_r(name);
      }
      virtual std::vector<size_t> dims_i(const std::string& name) const {
        return vc1.contains_r(name) ? vc1.dims_i(name) : vc2.dims_i(name);
      }
      virtual void names_r(std::vector<std::string>& names) const {
        vc1.names_r(names);
        std::vector<std::string> names2;
        vc2.names_r(names2);
        names.insert(names.end(), names2.begin(), names2.end());
      }
      virtual void names_i(std::vector<std::string>& names) const {
        vc1.names_i(names);
        std::vector<std::string> names2;
        vc2.names_i(names2);
        names.insert(names.end(), names2.begin(), names2.end());
      }

    };
  }
}

#endif
