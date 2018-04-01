add_subdirectory(c)

configure_file(config.pxi.in config.pxi)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package( PythonLibs REQUIRED )
include(UseCython)


include_directories(
  .
  c
  )

# add the command to generate the source code
# add_custom_command (
#   OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/Table.h
#   COMMAND MakeTable ${CMAKE_CURRENT_BINARY_DIR}/Table.h
#   DEPENDS MakeTable
#   )
 
add_custom_command(
  OUTPUT "aps_ui.py"
  COMMAND pyside-uic ${CMAKE_CURRENT_SOURCE_DIR}/ui/aps_ui.ui -o ${CMAKE_CURRENT_SOURCE_DIR}/aps_ui.py
  DEPENDS "ui/aps_ui.ui"
  )
add_custom_target(ui ALL DEPENDS "aps_ui.py")


set_source_files_properties(wrappers.c PROPERTIES COMPILE_FLAGS "${CMAKE_C_FLAGS} ${CYTHON_EXTRA_C_FLAGS}")  
cython_add_module(wrappers wrappers.pyx)
target_link_libraries(wrappers lasp_lib )


