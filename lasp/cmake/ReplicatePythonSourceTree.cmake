# Note: when executed in the build dir, then CMAKE_CURRENT_SOURCE_DIR is the
# build dir.
file( COPY src/pyx/ DESTINATION "${CMAKE_ARGV3}"
  FILES_MATCHING PATTERN "*.py" )
file( COPY src/pyx/ DESTINATION "${CMAKE_ARGV3}"
  FILES_MATCHING PATTERN "*.so" )
