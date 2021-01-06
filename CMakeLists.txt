cmake_minimum_required(VERSION 3.0.0)
project(Thesis VERSION 0.1.0)

include(CTest)
enable_testing()


find_package (Eigen3 3.3 REQUIRED)

include_directories("/usr/include/EigenRand-0.3.1/")
# include_directories("/opt/intel/oneapi/mkl/2021.1.1/include/")

set (MKLROOT $ENV{MKLROOT})

# message("$ENV{MKLROOT}")
find_package(OpenMP)
if (OPENMP_FOUND)
    # set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg -march=native -O2 -DMKL_ILP64  -m64  -I\"/opt/intel/oneapi/mkl/2021.1.1/include/\"")

SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_gnu_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lgomp -lpthread -lm -ldl")
# SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg")



set(CMAKE_MODULE_PATH "/root/thesis;${CMAKE_MODULE_PATH}")
# find_package(MKL)

add_executable(Thesis main.cpp m_matrix.h m_matrix.cpp hybrid_mc.h hybrid_mc.cpp helper_funcs.h helper_funcs.cpp)


target_link_libraries (Thesis Eigen3::Eigen)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
