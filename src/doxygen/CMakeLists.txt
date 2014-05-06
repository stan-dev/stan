set( doxyfile_in          ${CMAKE_CURRENT_SOURCE_DIR}/doxygen.cfg.in   )
set( doxyfile             ${CMAKE_CURRENT_BINARY_DIR}/docygen.cfg      )
set( doxy_html_index_file ${CMAKE_CURRENT_BINARY_DIR}/html/index.html )
set( doxy_extra_files     ${CMAKE_CURRENT_SOURCE_DIR}/Eigen.dox
                          ${CMAKE_CURRENT_SOURCE_DIR}/boost.dox     
                          ${CMAKE_CURRENT_SOURCE_DIR}/error_handling.dox     
                          ${CMAKE_CURRENT_SOURCE_DIR}/mainpage.dox     
                          ${CMAKE_CURRENT_SOURCE_DIR}/stan.dox     
                          ${CMAKE_CURRENT_SOURCE_DIR}/std.dox
                          ${PROJECT_SOURCE_DIR}/src/stan.hpp     )

file(GLOB_RECURSE STAN_SRC_FILES ${PROJECT_SOURCE_DIR}/src/stan/*.[ch]pp)

configure_file( ${doxyfile_in} ${doxyfile} @ONLY )

ADD_CUSTOM_COMMAND( OUTPUT ${doxy_html_index_file}
                    COMMAND ${DOXYGEN_EXECUTABLE} ${doxyfile}
                    DEPENDS ${doxyfile} ${doxy_extra_files} ${STAN_SRC_FILES}
                    COMMENT "Generating HTML documentation" )
 
ADD_CUSTOM_TARGET( doxygen ALL DEPENDS ${doxy_html_index_file} )
 
