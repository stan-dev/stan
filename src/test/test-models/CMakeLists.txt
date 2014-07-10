# stanc
set(STANC_SRCS stanc.cpp)
add_executable(stanc-bin ${STANC_SRCS})
set_target_properties(stanc-bin PROPERTIES OUTPUT_NAME stanc)
target_link_libraries(stanc-bin stanc)
set(STANC_BIN ${CMAKE_CURRENT_BINARY_DIR}/stanc)


add_subdirectory(no-main)
add_subdirectory(syntax-only)

