cmake_minimum_required(VERSION 3.8)

project(googletest-download NONE)

include(ExternalProject)
ExternalProject_Add(googletest
  GIT_REPOSITORY    https://github.com/google/googletest.git
  GIT_TAG           main
  SOURCE_DIR        "/home/ai8111/recovery/GOMC/build_test/googletest-src"
  BINARY_DIR        "/home/ai8111/recovery/GOMC/build_test/googletest-build"
  CONFIGURE_COMMAND ""
  BUILD_COMMAND     ""
  INSTALL_COMMAND   ""
  TEST_COMMAND      ""
)
