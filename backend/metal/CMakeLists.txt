add_library(kansai_backend_metal STATIC
  src/MetalOps.mm
)
target_include_directories(kansai_backend_metal PUBLIC include)
target_compile_features(kansai_backend_metal PUBLIC cxx_std_17)
set_source_files_properties(src/MetalOps.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")

find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
target_link_libraries(kansai_backend_metal PUBLIC ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
