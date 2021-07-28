#!/usr/bin/env bash

# sh initialize_venv.sh -f 'John Smith' -a 25 -u john
#
# The colons in the optstring mean that values are required for the
# corresponding flags. In the above example of u:d:p:f:, all flags are
# followed by a colon. This means all flags need a value. If, for example,
# the d and f flags were not expected to have a value, u:dp:f would be the
# optstring.

while getopts hiea: flag
do
    case "${flag}" in
        h) u_help=1;;
        i) u_install=1;;
        e) u_example=1;;
        a) u_path=${OPTARG};;
        *) u_invalid=1;;
    esac
done
if ["${u_invalid}" -eq "1"]; then
    echo "Invalid option: -$flag";
    return;
fi
echo "u_help: ${u_help}";
echo "u_install: ${u_install}";
echo "u_example: ${u_example}";
echo "u_path: ${u_path}";

#FILE=venv/bin/activate
#if [ -f "$FILE" ]; then
#    echo "deactivating any previous venv"
#    deactivate
#
#    echo "$FILE exists, sourcing it"
#    source ${FILE}
#
##    python3 -m pip install -U matplotlib
##    python3 -m pip install -U numpy
##    python3 -m pip install -U scipy
##    python3 -m pip install -U toopazo_tools
##    python3 -m pip install -U toopazo_tools
#
#    python -m toopazo_ulg.plot_main --bdir tests
#
#else
#    echo "$FILE does not exist."
#fi

# Case insensitive search and replace of 'foo' for 'bar' in 'hello.txt'
# sed -i 's/foo/bar/gI' hello.txt

# Case sensitive search and replace of 'foo' for 'bar' in 'hello.txt'
# sed -i 's/foo/bar/g' hello.txt

# Case sensitive search and replace in filenames .txt
# rename  's/foo/bat/' *.txt