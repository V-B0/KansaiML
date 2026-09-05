add_library(kansai_backend_cpu STATIC
  src/Ops.cpp
  src/Fused.cpp
)
target_include_directories(kansai_backend_cpu PUBLIC include)
target_compile_features(kansai_backend_cpu PUBLIC cxx_std_17)

if(APPLE)
  find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
  target_link_libraries(kansai_backend_cpu PUBLIC ${ACCELERATE_FRAMEWORK})
  target_compile_definitions(kansai_backend_cpu PUBLIC KANSAI_USE_ACCELERATE)
endif()
