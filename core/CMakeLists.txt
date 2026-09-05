set(KANSAI_CORE_SOURCES
  src/Storage.cpp
  src/StoragePool.cpp
  src/Tensor.cpp
  src/Fused.cpp
  src/GradOps.cpp
)
if(APPLE)
  list(APPEND KANSAI_CORE_SOURCES src/MetalOps.cpp)
endif()

add_library(kansai_core STATIC ${KANSAI_CORE_SOURCES})
target_include_directories(kansai_core PUBLIC include)
target_link_libraries(kansai_core PUBLIC kansai_backend_cpu)
target_compile_features(kansai_core PUBLIC cxx_std_17)

if(APPLE)
  target_link_libraries(kansai_core PUBLIC kansai_backend_metal)
  target_compile_definitions(kansai_core PUBLIC KANSAI_HAS_METAL)
endif()
