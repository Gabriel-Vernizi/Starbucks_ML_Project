@echo off

uv pip freeze > temp.txt

uv add -r temp.txt 

del temp.txt