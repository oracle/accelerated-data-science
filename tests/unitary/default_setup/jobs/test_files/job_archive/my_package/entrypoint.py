import os
from my_package import utils
from my_module import my_function_in_module

print("This is the entrypoint inside a package.")
my_function_in_module()
utils.my_function_in_package()
print(os.getcwd())
