## replace.py
# Description: Replace a string in a file
# Usage: replace(file_path, pattern, subst)
from tempfile import mkstemp
from shutil import move, copymode
from os import fdopen, remove

def replace(file_path, pattern, subst):
    fh, abs_path = mkstemp()
    with fdopen(fh,'w') as new_file, open(file_path) as old_file:
        new_file.write(old_file.read().replace(pattern, subst))
    copymode(file_path, abs_path)
    remove(file_path)
    move(abs_path, file_path)