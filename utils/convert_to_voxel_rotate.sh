#!/bin/bash

for filename in stl/*/*.STL; do
  ./binvox.exe -c -d $1 "$filename"
  ./binvox.exe -c -rotx -d $1 "$filename"
  ./binvox.exe -c -rotx -rotx -d $1 "$filename"
  ./binvox.exe -c -rotx -rotx -rotx -d $1 "$filename"
  ./binvox.exe -c -rotz -rotx -d $1 "$filename"
  ./binvox.exe -c -rotz -rotz -rotz -rotx -d $1 "$filename"
done
