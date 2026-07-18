if(NOT ACCELSIM_CONFIG)
  set(ACCELSIM_CONFIG "release")
endif()
if(ACCELSIM_CONFIG STREQUAL "debug" OR CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(GPGPUSIM_BUILD_TYPE "debug")
else()
  set(GPGPUSIM_BUILD_TYPE "release")
endif()

