#ifndef STAN__IO__ARRAY_VAR_CONTEXT_HPP
#define STAN__IO__ARRAY_VAR_CONTEXT_HPP

#include <map>
#include <vector>
#include <stan/io/var_context.hpp>

namespace stan {
  namespace io {
    
    /*
    namespace {
      size_t product(std::vector<size_t> dims) {
        size_t y = 1U;
        for (size_t i = 0; i < dims.size(); ++i)
          y *= dims[i];
        return y;
      }
    }
    */
    
    class array_var_context : public var_context {
    private:
      std::map<std::string, 
               std::pair<std::vector<double>,
                         std::vector<size_t> > > vars_r_;
      std::map<std::string, 
               std::pair<std::vector<int>, 
                         std::vector<size_t> > > vars_i_;
      std::vector<double> const empty_vec_r_;
      std::vector<int> const empty_vec_i_;
      std::vector<size_t> const empty_vec_ui_;

      bool contains_r_only(const std::string& name) const {
        return vars_r_.find(name) != vars_r_.end();
      }
      void validate(size_t num_par, size_t array_len, 
                    const std::vector<std::vector<size_t> >& dims) {
      }
      void add_r(const std::vector<std::string>& names, 
                 const std::vector<double>& values, 
                 const std::vector<std::vector<size_t> >& dims) {
        validate(names.size(), values.size(), dims);
        size_t start = 0;
        size_t end = 0;   
        for (size_t i = 0; i < names.size(); i++) {
          end += product(dims[i]);
          std::vector<double> v(values.begin() + start, values.begin() + end);
          vars_r_[names[i]] 
            = std::pair<std::vector<double>,
                        std::vector<size_t> >(v, dims[i]);
          start = end;
        }
      }
      void add_i(const std::vector<std::string>& names, 
                 const std::vector<int>& values, 
                 const std::vector<std::vector<size_t> >& dims) {
        validate(names.size(), values.size(), dims);
        size_t start = 0;
        size_t end = 0;   
        for (size_t i = 0; i < names.size(); i++) {
          end += product(dims[i]);
          std::vector<int> v(values.begin() + start, values.begin() + end);
          vars_i_[names[i]] 
            = std::pair<std::vector<int>,
                        std::vector<size_t> >(v, dims[i]);
          start = end;
        }
      }
  
    public:
      array_var_context(const std::vector<std::string>& names_r,
                        const std::vector<double>& values_r, 
                        const std::vector<std::vector<size_t> >& dim_r) {
        add_r(names_r, values_r, dim_r);
      }
      array_var_context(const std::vector<std::string>& names_i, 
                        const std::vector<int>& values_i, 
                        const std::vector<std::vector<size_t> >& dim_i) {
        add_i(names_i, values_i, dim_i);
      }
      array_var_context(const std::vector<std::string>& names_r, 
                        const std::vector<double>& values_r, 
                        const std::vector<std::vector<size_t> >& dim_r,
                        const std::vector<std::string>& names_i, 
                        const std::vector<int>& values_i, 
                        const std::vector<std::vector<size_t> >& dim_i) {
        add_r(names_r, values_r, dim_r);
        add_i(names_i, values_i, dim_i);
      }

      bool contains_r(const std::string& name) const {
        return contains_r_only(name) || contains_i(name);
      }
      bool contains_i(const std::string& name) const {
        return vars_i_.find(name) != vars_i_.end();
      }
      std::vector<double> vals_r(const std::string& name) const {
        if (contains_r_only(name)) {
          return (vars_r_.find(name)->second).first;
        } else if (contains_i(name)) {
          std::vector<int> vec_int = (vars_i_.find(name)->second).first;
          std::vector<double> vec_r(vec_int.size());
          for (size_t ii = 0; ii < vec_int.size(); ii++) {
            vec_r[ii] = vec_int[ii];
          }
          return vec_r;
        }
        return empty_vec_r_;
      }
      std::vector<size_t> dims_r(const std::string& name) const {
        if (contains_r_only(name)) {
          return (vars_r_.find(name)->second).second;
        } else if (contains_i(name)) {
          return (vars_i_.find(name)->second).second;
        }
        return empty_vec_ui_;
      }
      std::vector<int> vals_i(const std::string& name) const {
        if (contains_i(name)) {
          return (vars_i_.find(name)->second).first;
        }
        return empty_vec_i_;
      }
      std::vector<size_t> dims_i(const std::string& name) const {
        if (contains_i(name)) {
          return (vars_i_.find(name)->second).second;
        }
        return empty_vec_ui_;
      }
      virtual void names_r(std::vector<std::string>& names) const {
        names.resize(0);        
        for (std::map<std::string, 
                      std::pair<std::vector<double>,
                                std::vector<size_t> > >
                 ::const_iterator it = vars_r_.begin();
             it != vars_r_.end(); ++it)
          names.push_back((*it).first);
      }

      /**
       * Return a list of the names of the integer variables in
       * the dump.
       *
       * @param names Vector to store the list of names in.
       */
      virtual void names_i(std::vector<std::string>& names) const {
        names.resize(0);        
        for (std::map<std::string, 
                      std::pair<std::vector<int>, 
                                std::vector<size_t> > >
                 ::const_iterator it = vars_i_.begin();
             it != vars_i_.end(); ++it)
          names.push_back((*it).first);
      }

      /** 
       * Remove variable from the object.
       * 
       * @param name Name of the variable to remove.
       * @return If variable is removed returns <code>true</code>, else
       *   returns <code>false</code>.
       */
      bool remove(const std::string& name) {
        return (vars_i_.erase(name) > 0) 
          || (vars_r_.erase(name) > 0);
      }
 
    };
  }
}
#endif
