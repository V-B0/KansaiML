add_library(kansai_core STATIC
  src/Storage.cpp
  src/StoragePool.cpp
  src/Tensor.cpp
  src/Fused.cpp
  src/GradOps.cpp
)
target_include_directories(kansai_core PUBLIC include)
target_link_libraries(kansai_core PUBLIC kansai_backend_cpu)
target_compile_features(kansai_core PUBLIC cxx_std_17)
