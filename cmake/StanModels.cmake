# stan_model(TARGET STANFILE [NOMAIN] [NOBUILD])
#
# Parse (and potentially build) a Stan model file.
#
# TARGET is either an executable name or a variable name, see NOBUILD below.
# STANFILE should be the path to a Stan file.  It should be relative to
# ${CMAKE_CURRENT_SOURCE_DIR}.

# If NOBUILD is passed then the command to compile the model is created and
# ${TARGET} is set to the source file name.  Otherwise, an executable of the
# name TARGET is created which will compile the model.
#
# If NOMAIN is passed, stanc is called with --no_main option.
function(stan_model target stanfile)
  set(no_main OFF)
  set(do_build ON)
  string(REPLACE ".stan" ".cpp" srcfile ${stanfile})

  # Parse any extra keyword arguments
  foreach(word ${ARGN})
    if("${word}" STREQUAL NOMAIN)
      set(no_main ON)
    elseif("${word}" STREQUAL NOBUILD)
      set(do_build OFF)
    else()
      message(ERROR "Unrecognized stanmodel option: \"${word}\"")
    endif()
  endforeach()

  # Make the output directory
  get_filename_component(TARGET_DIR "${CMAKE_CURRENT_BINARY_DIR}/${srcfile}" DIRECTORY)
  file(MAKE_DIRECTORY "${TARGET_DIR}")


  # Construct stanc arguments
  set(stanc_args "${CMAKE_CURRENT_SOURCE_DIR}/${stanfile}"
                 --o="${CMAKE_CURRENT_BINARY_DIR}/${srcfile}")
  if(no_main)
    set(stanc_args ${stanc_args} --no_main)
  endif()

  # Add custom command to build the source file
  add_custom_command(OUTPUT ${srcfile}
     COMMAND ${STANC_BIN} ${stanc_args}
     DEPENDS stanc-bin)

  if(do_build)
    # Add the model executable
    add_executable(${target} ${srcfile})
    target_link_libraries(${target} stan)
  else()
    set(${target} "${srcfile}" PARENT_SCOPE)
  endif()
endfunction(stan_model)


