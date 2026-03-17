:: Runs the selected executable with the input file located the local directory
@echo off

set exe=C:\Users\valentinc\Workspace\OpenFAST\OpenFAST-mod\OpenFAST-mod-3.5.0\custom-build\glue-codes\openfast\openfast.exe

set dir=%cd%
cd %~dp0
for %%f in (*.fst) do (
%exe% %%f
)
cd %dir%