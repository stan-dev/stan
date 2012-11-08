#ifndef __STAN__IO__VAR_CONTEXT_HPP__
#define __STAN__IO__VAR_CONTEXT_HPP__

#include <vector>
#include <string>

namespace stan {

  namespace io {

    /**
     * A <code>var_reader</code> reads array variables of integer and
     * floating point type by name and dimension.  
     *
     * <p>An array's dimensionality is described by a sequence of
     * (unsigned) integers.  For instance, <code>(7, 2, 3)</code> picks
     * out a 7 by 2 by 3 array.  The empty dimensionality sequence
     * <code>()</code> is used for scalars.  
     *
     * <p>Multidimensional arrays are stored in column-major order,
     * meaning the first index changes the most quickly.
     *
     * <p>If a variable has integer variables, it should return
     * those integer values cast to floating point values when
     * accessed through the floating-point methods.
     */
    class var_context {
    public:

      /**
       * Return <code>true</code> if the specified variable name is
       * defined.  This method should return <code>true</code> even
       * if the values are all integers.
       *
       * @param name Name of variable.
       * @return <code>true</code> if the variable exists with real
       * values.
       */
      virtual bool contains_r(const std::string& name) const = 0;

      /**
       * Return the floating point values for the variable of the
       * specified variable name in last-index-major order.  This
       * method should cast integers to floating point values if the
       * values of the named variable are all integers.
       *
       * <p>If there is no variable of the specified name, the empty
       * vector is returned.
       *
       * @param name Name of variable.
       * @return Sequence of values for the named variable.
       */
      virtual std::vector<double> vals_r(const std::string& name) const = 0;

      /**
       * Return the dimensions for the specified floating point variable.
       * If the variable doesn't exist or if it is a scalar, the
       * return result should be the empty vector.
       *
       * @param name Name of variable.
       * @return Sequence of dimensions for the variable.
       */
      virtual std::vector<size_t> dims_r(const std::string& name) const = 0;

      /**
       * Return <code>true</code> if the specified variable name has
       * integer values.
       *
       * @param name Name of variable.
       * @return <code>true</code> if an integer variable of the specified
       * name is defined.
       */
      virtual bool contains_i(const std::string& name) const = 0;

      /**
       * Return the integer values for the variable of the specified
       * name in last-index-major order or the empty sequence if the
       * variable is not defined.
       *
       * @param name Name of variable.
       * @return Sequence of integer values.
       */
      virtual std::vector<int> vals_i(const std::string& name) const = 0;

      /**
       * Return the dimensions of the specified floating point variable.
       * If the variable doesn't exist (or if it is a scalar), the
       * return result should be the empty vector.
       *
       * @param name Name of variable.
       * @return Sequence of dimensions for the variable.
       */
      virtual std::vector<size_t> dims_i(const std::string& name) const = 0;

      /**
       * Return a list of the names of the floating point variables in
       * the context.
       *
       * @param names Vector to store the list of names in.
       */
      virtual void names_r(std::vector<std::string>& names) const = 0;

      /**
       * Return a list of the names of the integer variables in
       * the context.
       *
       * @param names Vector to store the list of names in.
       */
      virtual void names_i(std::vector<std::string>& names) const = 0;
    };
    

  }

}

#endif
