cmake_minimum_required(VERSION 3.15.0)

set(PROJECT "DualAnalysisPlugin")
PROJECT(${PROJECT})

include(FetchContent)
FetchContent_Declare(cmakesupport
	GIT_REPOSITORY https://github.com/ManiVaultStudio/github-actions
	GIT_TAG main
	GIT_SHALLOW TRUE
	SOURCE_SUBDIR CMakeSupport
)
FetchContent_MakeAvailable(cmakesupport)

# -----------------------------------------------------------------------------
# User options
# -----------------------------------------------------------------------------
option(USE_ARTIFACTORY_LIBS "Use the prebuilt libraries from artifactory" ON)
option(ENABLE_AVX "Enable AVX support" OFF)
option(MV_UNITY_BUILD "Combine target source files into batches for faster compilation" OFF)
set(OPTIMIZATION_LEVEL "2" CACHE STRING "Optimization level for all targets in release builds, e.g. 0, 1, 2")

# -----------------------------------------------------------------------------
# CMake options
# -----------------------------------------------------------------------------
set(VERBOSE TRUE)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# macros for linking both prebuilt artifactory libraries and locally built ones
include(CMakeLinkUtilities)

# Test hardware avx capabilities
include(CMakeSetOptimizationAndAVX)

include(InstallArtifactoryPackage)

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /DWIN32 /EHsc /MP  /permissive- /Zc:__cplusplus")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
endif(MSVC)

# -----------------------------------------------------------------------------
# Install dependency libraries
# -----------------------------------------------------------------------------

set(LIBRARY_INSTALL_DIR ${PROJECT_BINARY_DIR})
message(status "FetchContent: ${cmakesupport_SOURCE_DIR} ${cmakesupport_BINARY_DIR}")
include("${cmakesupport_SOURCE_DIR}/CMakeSupport/InstallArtifactoryPackage.cmake")
if (USE_ARTIFACTORY_LIBS AND NOT DUAL_ANALYSIS_ARTIFACTORY_LIBS_INSTALLED) 
    message(STATUS "Installing artifactory packages to: ${LIBRARY_INSTALL_DIR}")

    set(HDILib_VERSION 1.2.7)
    set(flann_VERSION 1.9.2)
    set(lz4_VERSION 1.9.3)

    # Both HDILib and flann are available prebuilt in the lkeb-artifactory as combined Debug/Release packages
    # lz4 is also available in the lkb-artifactory in separate Debug and |Release packages
    install_artifactory_package(HDILib ${HDILib_VERSION} biovault TRUE) 
    install_artifactory_package(flann ${flann_VERSION} lkeb TRUE)
    install_artifactory_package(lz4 ${lz4_VERSION} _ FALSE)

    message(STATUS "module path ${CMAKE_MODULE_PATH}")
    message(STATUS "Call find_package HDILib search ${CMAKE_SOURCE_DIR}/HDILib/lib/cmake/HDILib")
    message(STATUS "Run find_package HDILib")

    set(DUAL_ANALYSIS_ARTIFACTORY_LIBS_INSTALLED TRUE CACHE BOOL "Use the prebuilt libraries from artifactory" FORCE)
endif()

# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------

if(USE_ARTIFACTORY_LIBS AND DUAL_ANALYSIS_ARTIFACTORY_LIBS_INSTALLED)
    # If the HDILib_VERSION is not a x.y.z version number 
    # omit the version argument from find package
    find_package(HDILib COMPONENTS hdidimensionalityreduction hdiutils hdidata PATHS "${LIBRARY_INSTALL_DIR}/HDILib" CONFIG REQUIRED)
else()
    if(NOT HDILIB_ROOT)
        message(FATAL_ERROR "Define HDILIB_ROOT")
    endif()

    find_package(flann CONFIG REQUIRED)
    find_package(HDILib COMPONENTS hdidimensionalityreduction hdiutils hdidata PATHS "${HDILIB_ROOT}" CONFIG REQUIRED)

    message(STATUS "HDILib found at ${HDILIB_ROOT} with ${HDILib_LINK_LIBS}")    
endif()

find_package(Qt6 6.3.1 COMPONENTS Widgets WebEngineWidgets REQUIRED)

find_package(ManiVault COMPONENTS Core PointData ImageData CONFIG)

find_package(OpenGL REQUIRED)
find_package(OpenMP)

if(OpenMP_CXX_FOUND)
    message(STATUS "Found OpenMP")
endif()

if(UNIX)
    message(STATUS "pThreads for Linux")
    find_package(Threads REQUIRED)
endif(UNIX)

set(THIRD_PARTY_JSON
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/json/nlohmann/json.hpp
)

# Checks whether this is the first CMake run for this project.
# If so, sets the CMake Install Prefix to the ManiVault install directory.
# This if-statement must be placed after the project() command to work.
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX ${MV_INSTALL_DIR} CACHE PATH "..." FORCE)
endif()

# -----------------------------------------------------------------------------
# Source files
# -----------------------------------------------------------------------------

set(PLUGIN
    src/DualAnalysisPlugin.h
    src/DualAnalysisPlugin.cpp
	src/DualAnalysisPlugin.json
)

set(ACTIONS
    src/Actions/SettingsAction.h
    src/Actions/SettingsAction.cpp
)

set(COMMON
    src/Common/GradientDescentSettingsAction.h
	src/Common/GradientDescentSettingsAction.cpp
	src/Common/KnnParameters.h
	src/Common/KnnSettingsAction.h
	src/Common/KnnSettingsAction.cpp
	src/Common/OffscreenBuffer.h
	src/Common/OffscreenBuffer.cpp
	src/Common/TsneAnalysis.h
	src/Common/TsneAnalysis.cpp
	src/Common/TsneComputationAction.h
	src/Common/TsneComputationAction.cpp
	src/Common/TsneData.h
	src/Common/TsneParameters.h
)

set(TSNE
	src/tSNE/GeneralTsneSettingsAction.h
	src/tSNE/GeneralTsneSettingsAction.cpp
	src/tSNE/InitTsneSettings.h
	src/tSNE/InitTsneSettings.cpp
	src/tSNE/TsneSettingsAction.h
	src/tSNE/TsneSettingsAction.cpp
)

set(HSNE
    #src/HSNE/HsneAnalysisPlugin.h
	#src/HSNE/HsneAnalysisPlugin.cpp
    src/HSNE/GeneralHsneSettingsAction.h
	src/HSNE/GeneralHsneSettingsAction.cpp
	src/HSNE/HierarchyConstructionSettingsAction.h
	src/HSNE/HierarchyConstructionSettingsAction.cpp
	src/HSNE/HsneHierarchy.h
	src/HSNE/HsneHierarchy.cpp
	src/HSNE/HsneParameters.h
	src/HSNE/HsneRecomputeWarningDialog.h
	src/HSNE/HsneScaleAction.h
	src/HSNE/HsneScaleAction.cpp
	src/HSNE/HsneSettingsAction.h
	src/HSNE/HsneSettingsAction.cpp
)

set(SOURCES ${PLUGIN} ${ACTIONS} ${COMMON} ${TSNE} ${HSNE})

source_group(Plugin FILES ${PLUGIN})
source_group(Actions FILES ${ACTIONS})
source_group(Common FILES ${COMMON})
source_group(tSNE FILES ${TSNE})
source_group(HSNE FILES ${HSNE})


# -----------------------------------------------------------------------------
# CMake Target
# -----------------------------------------------------------
# Create dynamic library for the plugin
add_library(${PROJECT} SHARED ${SOURCES})

# -----------------------------------------------------------------------------
# Target include directories
# -----------------------------------------------------------------------------
# Include ManiVault headers, including system data plugins
target_include_directories(${PROJECT} PRIVATE "${ManiVault_INCLUDE_DIR}")
target_include_directories(${PROJECT} PRIVATE "src/Common")
target_include_directories(${PROJECT} PRIVATE "third_party/json")

set_HDILib_project_includes(${PROJECT})
set_flann_project_includes(${PROJECT})
set_lz4_project_includes(${PROJECT})

# -----------------------------------------------------------------------------
# Target properties
# -----------------------------------------------------------------------------
# Request C++17
target_compile_features(${PROJECT} PRIVATE cxx_std_17)

# Enable unity build
if(MV_UNITY_BUILD)
    set_target_properties(${PROJECT} PROPERTIES UNITY_BUILD ON)
endif()


# -----------------------------------------------------------------------------
# Target library linking
# -----------------------------------------------------------------------------
# Link to Qt libraries
target_link_libraries(${PROJECT} PRIVATE Qt6::Widgets)
target_link_libraries(${PROJECT} PRIVATE Qt6::WebEngineWidgets)

# Link to ManiVault and data plugins
target_link_libraries(${PROJECT} PRIVATE ManiVault::Core)
target_link_libraries(${PROJECT} PRIVATE ManiVault::PointData)
target_link_libraries(${PROJECT} PRIVATE ManiVault::ImageData)

target_link_libraries(${PROJECT} PRIVATE ${OPENGL_LIBRARIES})

if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT} PRIVATE OpenMP::OpenMP_CXX)
endif()

set_flann_project_link_libraries(${PROJECT})
set_HDILib_project_link_libraries(${PROJECT})
set_lz4_project_link_libraries(${PROJECT})

set_optimization_level(${PROJECT} ${OPTIMIZATION_LEVEL})
check_and_set_AVX(${PROJECT} ${ENABLE_AVX})

silence_opengl_deprecation(${PROJECT})


# -----------------------------------------------------------------------------
# Target installation
# -----------------------------------------------------------------------------
# Install the shared plugin libary to the "Plugins" folder in the ManiVault install directory
install(TARGETS ${PROJECT}
    RUNTIME DESTINATION Plugins COMPONENT PLUGINS # Windows .dll
    LIBRARY DESTINATION Plugins COMPONENT PLUGINS # Linux/Mac .so
)

add_custom_command(TARGET ${PROJECT} POST_BUILD
    COMMAND "${CMAKE_COMMAND}"
        --install ${PROJECT_BINARY_DIR}
        --config $<CONFIGURATION>
        --prefix ${ManiVault_INSTALL_DIR}/$<CONFIGURATION>
)

# -----------------------------------------------------------------------------
# Miscellaneous
# -----------------------------------------------------------------------------
# Automatically set the debug environment (command + working directory) for MSVC
if(MSVC)
    set_property(TARGET ${PROJECT} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY $<IF:$<CONFIG:DEBUG>,${ManiVault_INSTALL_DIR}/debug,${ManiVault_INSTALL_DIR}/release>)
    set_property(TARGET ${PROJECT} PROPERTY VS_DEBUGGER_COMMAND $<IF:$<CONFIG:DEBUG>,${ManiVault_INSTALL_DIR}/debug/ManiVault\ Studio.exe,${ManiVault_INSTALL_DIR}/release/ManiVault\ Studio.exe>)
endif()
