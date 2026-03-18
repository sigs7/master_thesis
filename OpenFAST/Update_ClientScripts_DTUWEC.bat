@echo on
cd %~dp0
set install_dir=DTUWEC-SC-mpi

call :Clone
call :Update
call :Build
goto :eof

:Clone
rmdir /s /q %install_dir%
git clone --branch master https://github.com/ValentinChb/DTUWEC.git %install_dir%
cd %install_dir%
git tag 0.0
rmdir /s /q utils
git clone --branch master https://gitlab.windenergy.dtu.dk/OpenLAC/utils.git utils
goto :eof

:Update
copy /y %~dp0%install_dir%\CMakeLists_utils.txt %~dp0%install_dir%\utils\cmake\CMakeLists.txt
goto :eof

:Build
cd %install_dir%
rmdir /s /q custom-build
mkdir custom-build
cd custom-build
cmake .. -G "MinGW Makefiles" -D CMAKE_Fortran_COMPILER="gfortran" -D CMAKE_BUILD_TYPE="release" -D CMAKE_LINK2SC=OFF
mingw32-make all
cd %~dp0
goto :eof
