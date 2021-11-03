# import os
# import subprocess

"""
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="toopazo_tools", # Replace with your own username
    version="0.0.17",
    author="toopazo",
    author_email="toopazo@protonmail.com",
    description="Python methods for common file and folder operations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/toopazo/toopazo_tools",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
"""


def increment_setup_version():
    filename = 'setup.py'

    # Read file
    fd = open(filename, 'r')
    line_arr = fd.readlines()
    fd.close()

    # Search for the specific line
    count = 0
    version = None
    version_line = None
    for i in range(0, len(line_arr)):
        line = line_arr[i]
        count += 1
        # print("line{}: {}".format(count, line))

        pattern0 = "version=\"0.0."
        pattern1 = "\","
        if pattern0 in line:
            print("[find_version] line {}: {}".format(count, line))
            version = line
            version = version.strip()
            version = version.replace(pattern0, "")
            version = version.replace(pattern1, "")
            version = int(version)
            version_line = i
            print("[find_version] version {}".format(version))
            print("[find_version] version_line {}".format(version_line))

    # Modify specific line
    orig_line = line_arr[version_line]
    new_line = orig_line.replace(str(version), str(version + 1))
    line_arr[version_line] = new_line

    # Write everythin again
    with open(filename, 'w') as file:
        file.writelines(line_arr)


# def increment_version(nversion):
#     # bash_command = "cwm --rdf test.rdf --ntriples > test.nt"
#     # sed -i 's/original/new/g' file.txt
#     pattern0 = "version=\"0.0."
#     pattern1 = "\","
#     orig = pattern0 + str(nversion) + pattern1
#     new = pattern0 + str(nversion + 1) + pattern1
#     bash_command = "sed -i 's/%s/%s/g' setup.py" % (orig, new)
#     print('[exec_cmd] bash_command %s' % bash_command)
#     process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
#     output, error = process.communicate()
#     print('[exec_cmd] error %s, output %s' % (error, output))


if __name__ == "__main__":
    increment_setup_version()
