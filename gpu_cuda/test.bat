@echo off
cls
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "CUDA_FILE=test.cu"
set "EXE=test.exe" 

set "VS_BUILD=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build"

REM Initialize Visual Studio environment for x64
call "%VS_BUILD%\vcvarsall.bat" x64



"%CUDA_PATH%\bin\nvcc" -c "%CUDA_FILE%" -o test.obj -Wno-deprecated-gpu-targets

if errorlevel 1 goto :Error



cl /Ox /EHsc /std:c++17 ^
  test.obj ^
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