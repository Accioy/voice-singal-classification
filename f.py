import os   
import platform

a=platform.platform()
if "Windows" in a:
    print("w")
elif "Linux" in a:
    print("l")
print(a)