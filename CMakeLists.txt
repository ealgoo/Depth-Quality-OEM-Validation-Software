# ubuntu 16.04 LTS cmake version 3.5.1
cmake_minimum_required(VERSION 2.8.3)

project(Depth_Quality_OEM_Validation_Software)

# Save the command line compile commands in the build output
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
endif()

find_package(OpenGL)
if(NOT OPENGL_FOUND)
    message(FATAL_ERROR "\n\n OpenGL package is missing!\n\n")
endif()

set(APP_NAME "Depth_Quality_OEM_Validation_Software")

if(UNIX)
    find_package(PkgConfig REQUIRED)

    find_package(OpenCV REQUIRED)
    find_package(X11 REQUIRED)
    find_package(Threads REQUIRED)
    pkg_check_modules(GTK3 REQUIRED gtk+-3.0)

    include_directories(${GTK3_INCLUDE_DIRS})
    link_directories(${GTK3_LIBRARY_DIRS})
    add_definitions(${GTK3_CFLAGS_OTHER})

    add_definitions(-DNOC_FILE_DIALOG_GTK)
elseif(WIN32)
    add_definitions(-DNOC_FILE_DIALOG_WIN32)
	#set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
	#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")
elseif(APPLE)
    add_definitions(-DNOC_FILE_DIALOG_OSX)
endif()


set(RS_IQC_CPP 
		D400-IQC.cpp 
#            include/model-views-IQC.h 
		model-views-IQC.cpp
		third-party/imgui/imgui.cpp 
		third-party/imgui/imgui_draw.cpp 
		third-party/imgui/imgui_impl_glfw.cpp 
		third-party/imgui/imgui-fonts-karla.hpp
		third-party/imgui/imgui-fonts-fontawesome.hpp
#           include/rendering.h 
	)

# config-ui
if(WIN32)
	add_executable(${APP_NAME} WIN32
		${RS_IQC_CPP} ${CMAKE_CURRENT_SOURCE_DIR}/res/resource.h
		${RS_IQC_CPP} ${CMAKE_CURRENT_SOURCE_DIR}/res/realsense-viewer.rc)
		
	source_group("Resources" FILES
		${CMAKE_CURRENT_SOURCE_DIR}/res/resource.h
		${CMAKE_CURRENT_SOURCE_DIR}/res/realsense-viewer.rc
		)
	include_directories(${APP_NAME} third-party third-party/imgui include/librealsense2 third-party/glfw/include include third-party/glfw/include ${CMAKE_CURRENT_SOURCE_DIR}/res/ third-party/opencv)
	
	math(EXPR BITS "8*${CMAKE_SIZEOF_VOID_P}")
	find_library(OPENCV_RELEASE_LIBRARY NAMES opencv_world330.lib HINTS ${CMAKE_CURRENT_SOURCE_DIR}/third-party/opencv/libs/${BITS})
	find_library(OPENCV_DEBUG_LIBRARY NAMES opencv_world330d.lib HINTS ${CMAKE_CURRENT_SOURCE_DIR}/third-party/opencv/libs/${BITS})
	find_library(GLFW_LIBRARY NAMES glfw3.lib HINTS ${CMAKE_CURRENT_SOURCE_DIR}/lib/${BITS})
	find_library(LIBRS_LIBRARY NAMES realsense2.lib HINTS ${CMAKE_CURRENT_SOURCE_DIR}/lib/${BITS})
	SET(OPENCV_LIBRARY optimized ${OPENCV_RELEASE_LIBRARY} debug ${OPENCV_DEBUG_LIBRARY})
	target_link_libraries(${APP_NAME} ${DEPENDENCIES} ${GTK3_LIBRARIES} ${OPENCV_LIBRARY} ${GLFW_LIBRARY} ${LIBRS_LIBRARY} ${OPENGL_LIBRARIES})

	add_custom_command(TARGET ${APP_NAME}
		POST_BUILD
		COMMAND if \"$(ConfigurationName)\" == \"Release\" (
			xcopy /d /y \"$(ProjectDir)..\\third-party\\opencv\\dlls\\$(PlatformArchitecture)\\opencv_world330.dll\" \"$(OutDir)\")
		COMMAND if \"$(ConfigurationName)\" == \"Debug\" (
			xcopy /d /y \"$(ProjectDir)..\\third-party\\opencv\\dlls\\$(PlatformArchitecture)\\opencv_world330d.dll\" \"$(OutDir)\")
		COMMAND xcopy /d /y \"$(ProjectDir)..\\config.json\" \"$(OutDir)\"
		COMMAND xcopy /d /y \"$(ProjectDir)..\\dll\\$(PlatformArchitecture)\\realsense2.dll\" \"$(OutDir)\"
	)

else()
        find_path(GLFW_INCLUDE_DIR NAMES GLFW/glfw3.h
            PATHS /usr/X11R6/include
                  /usr/include/X11
                  /opt/graphics/OpenGL/include
                  /opt/graphics/OpenGL/contrib/libglfw
                  /usr/local/include
                  /usr/include/GL
                  /usr/include
        )
        # Find glfw library
        find_library(GLFW_LIBRARIES NAMES glfw glfw3
                PATHS /usr/lib64
                      /usr/lib
                      /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}
                      /usr/local/lib64
                      /usr/local/lib
                      /usr/local/lib/${CMAKE_LIBRARY_ARCHITECTURE}
                      /usr/X11R6/lib
        )
	add_executable(${APP_NAME} 
		${RS_IQC_CPP})
	include_directories(${APP_NAME} third-party/imgui third-party include)
        target_link_libraries(${APP_NAME} ${GTK3_LIBRARIES} ${OpenCV_LIBS} ${X11_LIBRARIES} ${GLFW_LIBRARIES} realsense2 ${OPENGL_LIBRARIES} pthread)
	file(COPY config.json DESTINATION ${CMAKE_BINARY_DIR})
endif()

install(
	TARGETS

	${APP_NAME}

	RUNTIME DESTINATION bin
)

