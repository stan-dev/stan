if(MINGW)
  # Disable optimization for building the grammars 
  foreach(flag_var
          CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
          CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
    string(REGEX REPLACE "-O[^0]" "${empty_string}" ${flag_var} "${${flag_var}}")
    set(${flag_var} "${${flag_var}} -O2")
  endforeach(flag_var)
else()
  # Disable optimization for building the grammars 
  foreach(flag_var
          CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
          CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
    string(REGEX REPLACE "-O[^0]" "-O0" ${flag_var} "${${flag_var}}")
    string(REGEX REPLACE "/O[^0]" "/O0" ${flag_var} "${${flag_var}}")
  endforeach(flag_var)
endif()

file(GLOB_RECURSE LIBSTANC_SRCS *.cpp)

add_library(stanc STATIC ${LIBSTANC_SRCS})

