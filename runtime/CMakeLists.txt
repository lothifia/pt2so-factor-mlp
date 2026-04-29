cmake_minimum_required(VERSION 3.18)

project(pt2so_factor_mlp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Torch REQUIRED)
find_package(OpenSSL REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(PT2SO_BLOB_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/blob.cpp")
if(NOT EXISTS "${PT2SO_BLOB_SOURCE}")
    set(PT2SO_BLOB_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/blob_empty.cpp")
endif()

add_library(model SHARED
    ${PT2SO_BLOB_SOURCE}
    src/model_api.cpp
)

target_include_directories(model
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(model
    PRIVATE
        ${TORCH_LIBRARIES}
        OpenSSL::Crypto
)
