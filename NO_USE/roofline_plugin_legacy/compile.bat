@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

set CUPTI_DIR=D:\NVIDIA CUDA Toolkit 12.9\extras\CUPTI
set CUDA_DIR=D:\NVIDIA CUDA Toolkit 12.9
set INC=-I"%CUPTI_DIR%\include" -I"%CUPTI_DIR%\samples\common" -I"%CUPTI_DIR%\samples\extensions\include\profilerhost_util" -I"%CUPTI_DIR%\samples\extensions\include\c_util" -I"%CUDA_DIR%\include"
set NV_LIBS=-L"%CUPTI_DIR%\lib64" -L"%CUDA_DIR%\lib\x64" -lcupti -lnvperf_host -lnvperf_target -lcuda -lcudart -lnvml

echo Compiling Roofline Plugin DLL...
nvcc --shared -o roofline_plugin.dll roofline_plugin.cpp Eval.cpp Metric.cpp %INC% %NV_LIBS%
echo Compilation finished.
