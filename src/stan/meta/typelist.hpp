#ifndef STAN__META__TYPELIST_HPP
#define STAN__META__TYPELIST_HPP

namespace stan {

  namespace meta {

    /**
     * An empty type list.  Used to terminate non-empty type lists,
     * which are encoded with the class <code>typelist</code>.
     *
     * <p>This struct is the analogue of the "nil" constant in LISP.
     *
     * <p>See Alexandrescu's C++ template book for details or
     * his Dr. Dobbs article, "Generic Programming:Typelists and Applications", 
     * http://www.drdobbs.com/generic-programmingtypelists-and-applica/184403813
     */
    struct nil {
    }; 

    /**
     * A non-empty type list composed of a type and a type list.
     * Lists are terminated using <code>stan::meta::null_typelist</code>.
     *
     * <p>This struct is the analogue of the "cons" operation in LISP,
     * with the <code>head</code> providing the head of the list (what
     * the "car" operation would return in LISP) and <code>tail</code>
     * providing the tail type (what the "cdr" operation would
     * return in LISP).
     *
     * <p>See Alexandrescu's C++ template book for details or
     * his Dr. Dobbs article, "Generic Programming:Typelists and Applications", 
     * http://www.drdobbs.com/generic-programmingtypelists-and-applica/184403813
     *
     * @tparam T first type in list.
     * @tparam L rest of type list.
     */
    template <typename T, typename L>
    struct cons {

      /**
       * Typedef for the head of the list, template parameter <code>T</code>.
       */
      typedef T head;

      /**
       * Typedef for the tail of the list, template parameter <code>L</code>.
       */
      typedef L tail;
    };

    /**
     * A no-op struct to use as a default in typelists to suppress the
     * not-enough template arguments gripe that occurs otherwise.
     */
    struct dummy {
    };

    /**
     * A utility template class to simplify writing down type lists
     * with five typenames.  
     *
     * <p>This is the primary template class, and there are
     * specializations for one to four parameters.  
     *
     * All of the template parameters default to <code>dummy</code>, a
     * dummy class defined in <code>stan::meta</code> to act as
     * default values for list templates.
     *
     * @tparam T1 first typename in result list.
     * @tparam T2 second typename.
     * @tparam T3 third typename.
     * @tparam T4 fourth typename.
     * @tparam T5 fifth typename.
     */
    template <typename T1 = dummy, typename T2 = dummy,
              typename T3 = dummy, typename T4 = dummy, 
              typename T5 = dummy>
    struct typelist {

      /**
       * Typedef for the type list consisting of typenames
       * <code>T1</code> through <code>T5</code>.
       */
      typedef cons<T1,cons<T2,cons<T3,cons<T4,cons<T5,nil> > > > > type;

    };


    /**
     * Utility template specialization to simplify writing down type
     * lists with four typenames.
     *
     * @tparam T1 first typename in result list.
     * @tparam T2 second typename.
     * @tparam T3 third typename.
     * @tparam T4 fourth typename.
     */
    template <typename T1, typename T2, typename T3, typename T4>
    struct typelist<T1,T2,T3,T4,dummy> {

      /**
       * Typedef for the type list consisting of typenames
       * <code>T1</code> through <code>T4</code>.
       */
      typedef cons<T1,cons<T2,cons<T3,cons<T4,nil> > > > type;

    };



    /**
     * Utility template specialization to simplify writing down type
     * lists with three typenames.
     *
     * @tparam T1 first typename in result list.
     * @tparam T2 second typename.
     * @tparam T3 third typename.
     */
    template <typename T1, typename T2, typename T3>
    struct typelist<T1,T2,T3,dummy,dummy> {

      /**
       * Typedef for the type list consisting of typenames
       * <code>T1</code> through <code>T3</code>.
       */
      typedef cons<T1,cons<T2,cons<T3,nil> > > type;

    };


    /**
     * Utility template specialization to simplify writing down type
     * lists with two typenames.
     *
     * @tparam T1 first typename in result list.
     * @tparam T2 second typename.
     */
    template <typename T1, typename T2>
    struct typelist<T1,T2,dummy,dummy,dummy> {

      /**
       * Typedef for the type list consisting of typenames
       * <code>T1</code> through <code>T2</code>.
       */
      typedef cons<T1,cons<T2,nil> > type;

    };


    /**
     * Utility template specialization to simplify writing down type
     * lists with one typename.
     *
     * @tparam T1 first typename in result list.
     */
    template <typename T1>
    struct typelist<T1,dummy,dummy,dummy,dummy> {

      /**
       * Typedef for the type list consisting of typename
       * <code>T1</code>.
       */
      typedef cons<T1,nil> type;

    };

    /**
     * Utility template specialization to simplify writing down empty
     * type lists.
     */
    template <>
    struct typelist<dummy,dummy,dummy,dummy,dummy> {

      /**
       * Typedef for the empty type list.
       */
      typedef nil type;

    };


  }

}

#endif

