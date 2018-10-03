
configure_file(config.pxi.in config.pxi)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(PythonLibs REQUIRED )
include(UseCython)
find_package(Numpy REQUIRED )

include_directories(
   ${PYTHON_NUMPY_INCLUDE_DIR}
   .
   c
 )
 add_subdirectory(c)
 add_subdirectory(device)

# add the command to generate the source code
# add_custom_command (
#   OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/Table.h
#   COMMAND MakeTable ${CMAKE_CURRENT_BINARY_DIR}/Table.h
#   DEPENDS MakeTable
#   )

set(ui_files ui_apsrtsettings ui_mainwindow ui_figure ui_about
ui_apswidget ui_revtimewidget ui_slmwidget ui_daqwidget ui_apssettings
ui_figuredialog)
foreach(fn ${ui_files})
  add_custom_command(
    OUTPUT "${fn}.py"
    COMMAND pyside-uic ${CMAKE_CURRENT_SOURCE_DIR}/ui/${fn}.ui -o ${CMAKE_CURRENT_SOURCE_DIR}/${fn}.py
    DEPENDS "ui/${fn}.ui"
    )
    add_custom_target(${fn} ALL DEPENDS "${fn}.py")
endforeach(fn)

add_custom_command(
    OUTPUT "${PROJECT_SOURCE_DIR}/resources_rc.py"
    COMMAND pyside-rcc -py3 ui/resources.qrc -o ${PROJECT_SOURCE_DIR}/resources_rc.py
    DEPENDS "ui/resources.qrc"
)
add_custom_target(resources_rc ALL DEPENDS "${PROJECT_SOURCE_DIR}/resources_rc.py")




set_source_files_properties(wrappers.c PROPERTIES COMPILE_FLAGS "${CMAKE_C_FLAGS} ${CYTHON_EXTRA_C_FLAGS}")
cython_add_module(wrappers wrappers.pyx)
target_link_libraries(wrappers lasp_lib )
