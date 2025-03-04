# Custom Game Engine Development

Welcome to the Custom Game Engine project! This document outlines the core components, development tools, and additional libraries used in building a game engine optimized for NVIDIA GPUs.

## Core Components

### Graphics API

- **Vulkan**: A modern, low-level graphics API that provides fine-grained control over GPU resources and is well-suited for high-performance applications.
- **OpenGL**: A more established API that is easier to learn and use, but with less control over the hardware compared to Vulkan.
- **DirectX**: If you're targeting Windows platforms, DirectX is a powerful option with extensive support for graphics and compute tasks.

### Compute API

- **CUDA**: Use CUDA for general-purpose parallel computing tasks that can benefit from the massive parallelism of NVIDIA GPUs.
- **OpenCL**: An alternative to CUDA that is cross-platform and can run on different types of hardware.

### Shader Languages

- **GLSL (OpenGL Shading Language)**: Used with OpenGL for writing vertex, fragment, and compute shaders.
- **HLSL (High-Level Shading Language)**: Used with DirectX for shader programming.
- **SPIR-V**: An intermediate language for Vulkan shaders, which can be generated from GLSL or HLSL.

### Physics Engine

- **Bullet Physics**: An open-source physics engine that can be integrated into your game engine for realistic physics simulations.
- **PhysX**: NVIDIA's physics engine that can leverage GPU acceleration for physics computations.

### Audio Engine

- **OpenAL**: A cross-platform audio API for rendering 3D sound.
- **FMOD**: A popular audio engine used in many commercial games, offering advanced audio features.

### Networking

- **Enet**: A simple and robust networking library for real-time applications.
- **RakNet**: A networking engine designed for games, providing features like object replication and remote procedure calls.

### Scripting Language

- **Lua**: A lightweight scripting language that can be embedded into your engine for game logic and customization.
- **Python**: Another option for scripting, though it may require more integration work.

## Development Tools

### Integrated Development Environment (IDE)

- **Visual Studio**: A powerful IDE for C++ development, especially on Windows.
- **CLion**: A cross-platform C++ IDE with good support for CMake projects.

### Version Control

- **Git**: Essential for managing your codebase and collaborating with others.

### Build System

- **CMake**: A cross-platform build system that can generate build files for various compilers and IDEs.

### Profiling and Debugging

- **NVIDIA Nsight**: A suite of tools for debugging and profiling GPU applications.
- **Valgrind**: A tool for memory debugging and profiling on Linux.

## Additional Libraries

### Math Libraries

- **GLM (OpenGL Mathematics)**: A header-only library for mathematics, designed for graphics applications.
- **Eigen**: A C++ template library for linear algebra.

### Image and Texture Loading

- **stb_image**: A simple library for loading images in various formats.
- **DevIL**: A more comprehensive image library for loading and manipulating images.

### UI Framework

- **Dear ImGui**: A bloat-free graphical user interface library for C++.
