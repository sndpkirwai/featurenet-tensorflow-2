#!/bin/bash

for filename in stl/*/*.STL; do
#  echo "`pwd`"
  echo "$filename"
#  # cd "$(dirname "${filename}")"
#  FILE="$(basename "${filename}")"
  chmod 755 binvox.exe
  ./binvox.exe -c -d 64 "$filename"
done
