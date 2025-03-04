@echo off
cls
REM ================================
REM Compiler and Environment Setup
REM -------------------------------
REM Set CUDA and Visual Studio paths
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "VS_BUILD=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build"

REM Initialize Visual Studio environment for x64
call "%VS_BUILD%\vcvarsall.bat" x64

REM ================================
REM Source Files
REM -------------------------------
set "OUT=cmd_output\output.cpp"
set "FILE=tools\bmpread.cpp"
set "GAME=game.cpp"
set "EXE=run.exe" 
set "CUDA_FILE=gpu_cuda\kernel.cu"


REM ================================
REM Compile CUDA file
REM -------------------------------
"%CUDA_PATH%\bin\nvcc" -c "%CUDA_FILE%" -o kernel.obj -Wno-deprecated-gpu-targets
if errorlevel 1 goto :Error

REM ================================
REM Compile and Link C++ files
REM -------------------------------
REM Notice the careful use of ^ for line continuation.
REM Make sure no extra ^ characters or trailing spaces exist.

cl /Ox /EHsc /std:c++17 ^
  "%OUT%" ^
  "%FILE%" ^
  "%GAME%" ^
  kernel.obj ^
  /Fe"%EXE%" /link /LIBPATH:"%CUDA_PATH%\lib\x64" cudart.lib

if errorlevel 1 goto :Error

echo Build successful.
goto :End

:Error
echo.
echo Build FAILED.
pause

:End
@echo on