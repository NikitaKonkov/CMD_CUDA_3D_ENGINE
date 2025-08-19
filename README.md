# CMD CUDA 3D Engine

A 3D ASCII-art rendering engine that uses CUDA for GPU-accelerated computation and renders textured 3D cubes directly to the Windows Command Prompt using ANSI escape sequences.

![Engine Demo](images/preview.gif) <!-- Add this later if you create a demo gif -->

## Overview

This project is a **3D ASCII-art renderer** that combines:
- **CUDA GPU computing** for high-performance 3D transformations and rasterization
- **Windows Console output** with ANSI color codes for real-time 3D visualization
- **BMP texture mapping** for applying textures to 3D cubes
- **Real-time input handling** for interactive camera movement

The engine renders 3D textured cubes as ASCII characters in the terminal, with each character colored using the 256-color ANSI palette based on the underlying texture.

## Features

- 🎮 **Real-time 3D Rendering**: GPU-accelerated 3D transformations and rasterization
- 🖼️ **Texture Mapping**: BMP image loading and texture application to 3D objects
- 🎨 **ASCII Art Output**: Converts 3D scenes to colored ASCII characters in the terminal
- ⌨️ **Interactive Controls**: WASD movement, arrow keys for camera control
- 🚀 **CUDA Acceleration**: Parallel processing for vertex transformations and pixel operations
- 🪟 **Windows Console**: Optimized for Windows Command Prompt with ANSI color support

## Technical Architecture

### Core Components

1. **CUDA Kernels** (`gpu_cuda/kernel.cu`):
   - Vertex transformation and rotation
   - Texture sampling and rasterization
   - Parallel pixel buffer operations

2. **BMP Reader** (`tools/bmpread.cpp`):
   - Custom BMP file parser
   - Texture data extraction for GPU upload

3. **Console Output** (`cmd_output/output.cpp`):
   - ANSI escape sequence generation
   - Optimized console buffer rendering
   - Real-time input handling

4. **Main Engine** (`game.cpp`):
   - Application entry point and main loop

### Rendering Pipeline

```
BMP Textures → CUDA Memory → 3D Transformations → Rasterization → ASCII Conversion → Console Output
```

## Prerequisites

- **Windows 10/11** (required for ANSI console support)
- **NVIDIA GPU** with CUDA support (Compute Capability 3.5+)
- **CUDA Toolkit 12.8** or compatible version
- **Visual Studio 2022** with C++ build tools
- **Windows Console** with ANSI escape sequence support enabled

## Installation & Build

1. **Clone the repository**:
   ```bash
   git clone https://github.com/NikitaKonkov/CMD_CUDA_3D_ENGINE.git
   cd CMD_CUDA_3D_ENGINE
   ```

2. **Run the build script**:
   ```batch
   install.bat
   ```
   
   This script will:
   - Set up the Visual Studio build environment
   - Compile CUDA kernels with `nvcc`
   - Link C++ source files with CUDA runtime
   - Generate the executable `run.exe`

3. **Run the engine**:
   ```batch
   run.exe
   ```

## Controls

| Key | Action |
|-----|--------|
| `W` | Move camera forward |
| `S` | Move camera backward |
| `A` | Move camera left |
| `D` | Move camera right |
| `Q` | Increase movement speed |
| `E` | Decrease movement speed |
| `↑` | Move object away |
| `↓` | Move object closer |
| `←` | Move object left |
| `→` | Move object right |

## Configuration

### Display Settings

Edit `header.h` to modify rendering resolution:

```cpp
const int WIDTH  = 238;  // Console width in characters
const int HEIGHT = 143;  // Console height in characters
```

### Texture Settings

Textures are loaded from the `images/` directory. The engine supports:
- **128x128 BMP files** for optimal performance
- **256-color palette** mapping to ANSI codes
- **Multiple texture formats** (see `images/` folder)

## Project Structure

```
CMD_CUDA_3D_ENGINE/
├── 📁 gpu_cuda/          # CUDA kernels and GPU code
│   ├── kernel.cu         # Main rendering kernels
│   ├── shapes.cu         # 3D shape definitions
│   └── test.cu           # CUDA testing utilities
├── 📁 cmd_output/        # Console output and input handling
│   └── output.cpp        # ANSI rendering and controls
├── 📁 tools/             # Utility functions
│   └── bmpread.cpp       # BMP file reader
├── 📁 images/            # Texture assets
│   ├── *.bmp             # Various texture files
│   └── 128xNUM/          # Numbered textures
├── header.h              # Main header with declarations
├── game.cpp              # Application entry point
├── install.bat           # Build script
└── README.md             # This file
```

## Performance

- **Resolution**: 238×143 characters (~34,000 pixels)
- **Frame Rate**: Depends on GPU performance and texture complexity
- **Memory Usage**: Optimized CUDA constant memory for textures
- **Console Output**: Efficient ANSI sequence batching

## Troubleshooting

### Build Issues

1. **CUDA not found**: Ensure CUDA Toolkit is installed and `CUDA_PATH` is correct in `install.bat`
2. **Visual Studio errors**: Verify VS 2022 Community is installed with C++ build tools
3. **Linking errors**: Check that `cudart.lib` is available in the CUDA lib directory

### Runtime Issues

1. **No output**: Enable ANSI support in Windows Console or use Windows Terminal
2. **Poor performance**: Ensure you have a CUDA-compatible NVIDIA GPU
3. **Texture errors**: Verify BMP files are in the correct format and path

## Development

### Adding New Textures

1. Convert images to 128×128 BMP format
2. Place in `images/` directory  
3. Update texture loading paths in `kernel.cu`

### Modifying 3D Objects

Edit the vertex and face definitions in `gpu_cuda/kernel.cu`:

```cpp
const Face h_CUBE_FACES[6] = {
    // Define your 3D object faces here
};
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with `install.bat`
5. Submit a pull request

## License

This project is open source. See LICENSE file for details.

## Acknowledgments

- NVIDIA CUDA for GPU computing framework
- Windows Console ANSI support for colored terminal output
- ASCII art rendering techniques and optimization strategies
