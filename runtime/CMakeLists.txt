cmake_minimum_required(VERSION 3.18)

project(pt2so_factor_mlp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(PT2SO_STATIC_OPENSSL "Statically link OpenSSL libcrypto into libmodel.so." ON)
set(PT2SO_LIBCRYPTO_A "" CACHE FILEPATH "Optional explicit path to libcrypto.a.")
set(PT2SO_OPENSSL_INCLUDE_DIR "" CACHE PATH "Optional explicit path to OpenSSL include directory.")

if(PT2SO_STATIC_OPENSSL)
    set(OPENSSL_USE_STATIC_LIBS TRUE)
endif()

find_package(Torch REQUIRED)
find_package(Threads REQUIRED)

if(PT2SO_LIBCRYPTO_A)
    if(NOT EXISTS "${PT2SO_LIBCRYPTO_A}")
        message(FATAL_ERROR "PT2SO_LIBCRYPTO_A does not exist: ${PT2SO_LIBCRYPTO_A}")
    endif()

    if(NOT PT2SO_OPENSSL_INCLUDE_DIR)
        find_path(PT2SO_OPENSSL_INCLUDE_DIR
            NAMES openssl/evp.h
            REQUIRED
        )
    endif()

    add_library(PT2SO::Crypto STATIC IMPORTED)
    set_target_properties(PT2SO::Crypto PROPERTIES
        IMPORTED_LOCATION "${PT2SO_LIBCRYPTO_A}"
        INTERFACE_INCLUDE_DIRECTORIES "${PT2SO_OPENSSL_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "${CMAKE_DL_LIBS};Threads::Threads"
    )
    set(PT2SO_CRYPTO_TARGET PT2SO::Crypto)
else()
    find_package(OpenSSL REQUIRED COMPONENTS Crypto)
    set(PT2SO_CRYPTO_TARGET OpenSSL::Crypto)

    if(PT2SO_STATIC_OPENSSL)
        get_target_property(PT2SO_OPENSSL_CRYPTO_LOCATION OpenSSL::Crypto IMPORTED_LOCATION)
        get_target_property(PT2SO_OPENSSL_CRYPTO_LOCATION_RELEASE OpenSSL::Crypto IMPORTED_LOCATION_RELEASE)
        if(PT2SO_OPENSSL_CRYPTO_LOCATION_RELEASE)
            set(PT2SO_OPENSSL_CRYPTO_LOCATION "${PT2SO_OPENSSL_CRYPTO_LOCATION_RELEASE}")
        endif()
        if(PT2SO_OPENSSL_CRYPTO_LOCATION AND NOT PT2SO_OPENSSL_CRYPTO_LOCATION MATCHES "\\.a$")
            message(FATAL_ERROR
                "PT2SO_STATIC_OPENSSL=ON but OpenSSL::Crypto did not resolve to libcrypto.a: "
                "${PT2SO_OPENSSL_CRYPTO_LOCATION}. Set PT2SO_LIBCRYPTO_A=/path/to/libcrypto.a."
            )
        endif()
    endif()
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

option(PT2SO_BUILD_MODEL_LIB "Build the embedded model shared library." ON)

if(PT2SO_BUILD_MODEL_LIB)
    set(PT2SO_BLOB_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/blob.cpp")
    if(NOT EXISTS "${PT2SO_BLOB_SOURCE}")
        set(PT2SO_BLOB_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/blob_empty.cpp")
    endif()

    set(PT2SO_KEY_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/aes_key_obfuscated.cpp")
    if(NOT EXISTS "${PT2SO_KEY_SOURCE}")
        set(PT2SO_KEY_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/aes_key_empty.cpp")
    endif()

    add_library(model SHARED
        ${PT2SO_BLOB_SOURCE}
        ${PT2SO_KEY_SOURCE}
        src/model_api.cpp
        src/model_session.cpp
    )

    target_include_directories(model
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    target_link_libraries(model
        PRIVATE
            ${TORCH_LIBRARIES}
            ${PT2SO_CRYPTO_TARGET}
    )

    if(UNIX AND NOT APPLE)
        target_link_options(model PRIVATE "LINKER:--exclude-libs,ALL")
    endif()
endif()
