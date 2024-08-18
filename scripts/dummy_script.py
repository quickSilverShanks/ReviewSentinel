'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Script Functionality
-- This script does nothing, just a test code to check functionality of click library to pass input parameters to script
Sample script execution commands:
    python scripts/dummy_script.py
    python scripts/dummy_script.py --param2 newfile1.py --param2 newfile2.py
    python scripts/dummy_script.py --param1 dummyval --param2 newfile1.py --param2 newfile2.py
    python scripts/dummy_script.py --param1 dummyval --param2 newfile1.py --param2 newfile2.py --param3
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

print("Script executed succesfully!!")

import click

@click.command()
@click.option(
    "--param1",
    default="p1 value",
    help="qwerty1"
)
@click.option(
    "--param2",
    default=("file1.py", "file2.py"),
    help="provide multiple file names",
    multiple=True
)
@click.option(
    "--param3",
    is_flag=True,
    help="boolean parameter"
)
def functn(param1, param2, param3):
    print(f"param1 : {param1}")
    print(f"param2, paramtype : {param2}, {type(param2)}")
    print(f"param2 : element1 : {param2[0]}")
    print(f"param2 : element2 : {param2[1]}")
    if param3:
        print(f"param3 True : {param3}")
    else:
        print(f"param3 False : {param3}")

if __name__=="__main__":
    functn()