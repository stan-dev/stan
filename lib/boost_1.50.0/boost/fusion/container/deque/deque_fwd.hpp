/*=============================================================================
    Copyright (c) 2005-2011 Joel de Guzman
    Copyright (c) 2005-2007 Dan Marsden

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
==============================================================================*/
#if !defined(FUSION_DEQUE_FORWARD_02092007_0749)
#define FUSION_DEQUE_FORWARD_02092007_0749

#include <boost/fusion/container/deque/limits.hpp>
#include <boost/preprocessor/repetition/enum_params_with_a_default.hpp>

///////////////////////////////////////////////////////////////////////////////
// With no decltype and variadics, we will use the C++03 version
///////////////////////////////////////////////////////////////////////////////
#if (defined(BOOST_NO_DECLTYPE)             \
  || defined(BOOST_NO_VARIADIC_TEMPLATES)   \
  || defined(BOOST_NO_RVALUE_REFERENCES))
# include <boost/fusion/container/deque/detail/cpp03_deque_fwd.hpp>
#else
#if defined(__WAVE__) && defined(BOOST_FUSION_CREATE_PREPROCESSED_FILES)
#pragma wave option(preserve: 2, line: 0, output: "detail/preprocessed/deque" FUSION_MAX_DEQUE_SIZE_STR "_fwd.hpp")
#endif

///////////////////////////////////////////////////////////////////////////////
// C++11 interface
///////////////////////////////////////////////////////////////////////////////
namespace boost { namespace fusion
{
    template <typename ...T>
    struct deque;
}}

#endif
#endif
