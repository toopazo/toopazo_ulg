# toopazo_ulg
Simple Python package that makes use pyulog commands
(see https://github.com/PX4/pyulog ) to: parse, process and plot
Px4 .ulg log files ( see https://github.com/PX4/Firmware )

Important links
- https://packaging.python.org/tutorials/packaging-projects/
- https://pypi.org/
- https://pypi.org/project/toopazo-ulg/

## How it works
It needs three folders: logs, tmp and plots (names are hardcoded).
It works by taking all .ulg files from the logs folder and then using
ulog2csv to writing the resulting .csv files in the tmp folder.
It finally writes all results in the plots folder.

## Dependencies
Install dependencies running pip as a module
```
 python -m pip install -U matplotlib
 python -m pip install -U numpy
 python -m pip install -U scipy
 python -m pip install -U toopazo_tools
```

## Usage
```
user@host: ls
logs tmp plots
user@host: python -m toopazo_ulg.plot_main --bdir tests/
```
