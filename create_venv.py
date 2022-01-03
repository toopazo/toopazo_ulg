#!/usr/bin/env python3

import subprocess
import argparse


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


def test_subprocess():
    cmd = ['python', '--`version']
    # result = subprocess.run(cmd, stdout=subprocess.PIPE)
    # result = subprocess.run(cmd, stdout=subprocess.PIPE,
    #                         stderr=subprocess.PIPE)
    result = subprocess.run(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    # result = subprocess.run(cmd, capture_output=True)
    # print('cmd %s result %s' % (cmd, result))
    print('result.stdout %s' % result.stdout)
    # cmd = 'ls /usr/bin/python'


def check_python3_version():
    for i in range(9, 6, -1):
        pver = 'python3.%s' % i
        cmd = [pver, '--version']
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT)
            # print('result.stdout %s' % result.stdout)
            _ = result
            return pver
        except FileNotFoundError:
            result = 'FileNotFoundError'
            # print('cmd %s result %s' % (cmd, result))
            _ = result
    return 'No python 3.x found'


def exec_cmd_and_report(cmd_str, decode):
    result = subprocess.run(
        cmd_str, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print('cmd %s' % cmd_str)
    print('result.stdout %s' % result.stdout)
    if decode:
        bstr = result.stdout
        result_str = bstr.decode()
        print('result_str')
        print(result_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Operations related to Python venv')
    parser.add_argument('--pver', action='store_true',
                        help='Get highest python3 version')
    parser.add_argument('--iacp', action='store_true',
                        help='Increment version, '
                             'add any changes, commit and push to repo')
    args = parser.parse_args()

    if args.pver:
        upver = check_python3_version()
        print(upver)

    if args.iacp:
        increment_setup_version()
        exec_cmd_and_report(['git', 'status'], decode=True)
        exec_cmd_and_report(['git', 'add', '.'], decode=True)
        exec_cmd_and_report(['git', 'commit', '-m',
                             '\"automated commit using iacp\"'], decode=True)
        exec_cmd_and_report(['git', 'push'], decode=True)
        exec_cmd_and_report(['git', 'status'], decode=True)

