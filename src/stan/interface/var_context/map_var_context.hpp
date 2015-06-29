#ifndef STAN_INTERFACE_VAR_CONTEXT_MAP_VAR_CONTEXT_HPP
#define STAN_INTERFACE_VAR_CONTEXT_MAP_VAR_CONTEXT_HPP

#include <map>
#include <pair>
#include <ostream>

#include <stan/interface/var_context/var_context.hpp>

namespace stan {
  namespace interface {
    namespace var_context {

      // Implements a var_context using an STL map to associate
      // parameter names with values and dimensions.
      class map_var_context : public var_context {
      private:
        
        typedef std::pair<std::vector<double>, std::vector<size_t> > vals_r_t;
        typedef std::map<std::string, vals_r_t> vars_r_t;
        
        typedef std::pair<std::vector<int>, std::vector<size_t> > vals_i_t;
        typedef std::map<std::string, vals_i_t> vars_i_t;
        
        vars_r_t vars_r_;
        vars_i_t vars_i_;
        
        std::vector<double> const empty_vec_r_;
        std::vector<int> const empty_vec_i_;
        std::vector<size_t> const empty_vec_ui_;
        
      public:
        
        map_var_context(const vars_r_t& vars_r, const vars_i_t& vars_i) {
          vars_r_ = vars_r;
          vars_i_ = vars_i;
        }
        
        /**
         * Return <code>true</code> if this dump contains the specified
         * variable name is defined. This method returns <code>true</code>
         * even if the values are all integers.
         *
         * @param name Variable name to test.
         * @return <code>true</code> if the variable exists.
         */
        bool contains_r(const std::string& name) const {
          return vars_r_.find(name) != vars_r_.end();
        }
        
        /**
         * Return <code>true</code> if this dump contains an integer
         * valued array with the specified name.
         *
         * @param name Variable name to test.
         * @return <code>true</code> if the variable name has an integer
         * array value.
         */
        bool contains_i(const std::string& name) const {
          return vars_i_.find(name) != vars_i_.end();
        }
        
        /**
         * Return the double values for the variable with the specified
         * name or null.
         *
         * @param name Name of variable.
         * @return Values of variable.
         */
        std::vector<double> vals_r(const std::string& name) const {
          if (contains_r(name))
            return (vars_r_.find(name)->second).first;
          return empty_vec_r_;
        }
        
        /**
         * Return the dimensions for the double variable with the specified
         * name.
         *
         * @param name Name of variable.
         * @return Dimensions of variable.
         */
        std::vector<size_t> dims_r(const std::string& name) const {
          if (contains_r(name))
            return (vars_r_.find(name)->second).second;
          return empty_vec_ui_;
        }
        
        /**
         * Return the integer values for the variable with the specified
         * name.
         *
         * @param name Name of variable.
         * @return Values.
         */
        std::vector<int> vals_i(const std::string& name) const {
          if (contains_i(name))
            return (vars_i_.find(name)->second).first;
          return empty_vec_i_;
        }
        
        /**
         * Return the dimensions for the integer variable with the specified
         * name.
         *
         * @param name Name of variable.
         * @return Dimensions of variable.
         */
        std::vector<size_t> dims_i(const std::string& name) const {
          if (contains_i(name))
            return (vars_i_.find(name)->second).second;
          return empty_vec_ui_;
        }
        
        /**
         * Return a list of the names of the floating point variables in
         * the dump.
         *
         * @param names Vector to store the list of names in.
         */
        void names_r(std::vector<std::string>& names) const {
          names.resize(0);
          for (vars_r_t::const_iterator it = vars_r_.begin();
               it != vars_r_.end(); ++it)
            names.push_back((*it).first);
        }
        
        /**
         * Return a list of the names of the integer variables in
         * the dump.
         *
         * @param names Vector to store the list of names in.
         */
       void names_i(std::vector<std::string>& names) const {
          names.resize(0);
          for (vars_i_t::const_iterator it = vars_i_.begin();
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
          return (vars_i_.erase(name) > 0) || (vars_r_.erase(name) > 0);
        }
          
        void add_r(const std::string& name,
                   const std::vector<double>& values,
                   const std::vector<size_t>& dims
                   std::ostream *o) {
          if (!constains_r(name)) {
            if (o)
              *o << name << " already exists as a real variable "
                 << "and will be not added." << std::endl;
            return;
          }
          vals_r_[name] = vals_r_t(values, dims);
        }
        
        void add_i(const std::string& name,
                   const std::vector<i>& values,
                   const std::vector<size_t>& dims
                   std::ostream *o) {
          if (!constains_i(name)) {
            if (o)
              *o << name << " already exists as an integer variable "
                 << "and will be not added." << std::endl;
            return;
          }
          vals_i_[name] = vals_i_t(values, dims);
        }
        
      };
      

    }
  }
}

#endif
