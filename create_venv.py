#!/usr/bin/env python3

import subprocess
import os


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
        python_ver = 'python3.%s' % i
        cmd = [python_ver, '--version']
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT)
            # print('result.stdout %s' % result.stdout)
            return python_ver
        except FileNotFoundError:
            result = 'FileNotFoundError'
            # print('cmd %s result %s' % (cmd, result))
    return 'No python 3.x found'


if __name__ == '__main__':
    pver = check_python3_version()
    print(pver)

