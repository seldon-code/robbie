#!/bin/sh 
find ./include -regex '.*\.\(cpp\|hpp\|cc\|cxx\)' -exec clang-format -i {} \;
find ./test -regex '.*\.\(cpp\|hpp\|cc\|cxx\)' -exec clang-format -i {} \;