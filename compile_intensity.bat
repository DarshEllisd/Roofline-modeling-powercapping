@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

echo Compiling intensity_kernels.cu to intensity_kernels.dll...
nvcc --shared -O3 -arch=sm_89 -o intensity_kernels.dll intensity_kernels.cu -lcudart
if %ERRORLEVEL% EQU 0 (
    echo Compilation succeeded!
) else (
    echo Compilation failed!
)
