# stan_model(TARGET STANFILE [NOMAIN] [NOBUILD] [CMDSTAN])
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
#
# If CMDSTAN is passed, then use the cmdstan binary instead.
function(stan_model target stanfile)
  set(no_main OFF)
  set(do_build ON)
  set(use_cmdstan OFF)
  string(REPLACE ".stan" ".cpp" srcfile ${stanfile})

  # Parse any extra keyword arguments
  foreach(word ${ARGN})
    if("${word}" STREQUAL NOMAIN)
      set(no_main ON)
    elseif("${word}" STREQUAL NOBUILD)
      set(do_build OFF)
    elseif("${word}" STREQUAL CMDSTAN)
      set(use_cmdstan ON)
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
  set(BUILD_SOURCES "${srcfile}")
  if (use_cmdstan)
    add_custom_command(OUTPUT ${srcfile}
       COMMAND ${RUNCMD} ${CMDSTANC_BIN} ${stanc_args}
       DEPENDS cmdstan-stanc)
    if (NOT no_main)
      set(BUILD_SOURCES "${BUILD_SOURCES}" "${CMDSTAN_MAIN_FILE}")
    endif()
  else()
    add_custom_command(OUTPUT ${srcfile}
       COMMAND ${RUNCMD} ${STANC_BIN} ${stanc_args}
       DEPENDS stanc-bin)
  endif()

  if(do_build)
    # Add the model executable
    add_executable(${target} ${BUILD_SOURCES})
    target_link_libraries(${target} stan)
  else()
    set(${target} "${srcfile}" PARENT_SCOPE)
  endif()
endfunction(stan_model)


