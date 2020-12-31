# toopazo_ulg
Simple Python package that makes use pyulog commands
(see https://github.com/PX4/pyulog ) to: parse, process and plot
Px4 .ulg log files ( see https://github.com/PX4/Firmware )

## How it works
It needs three folders: logs, tmp and plots (names are hardcoded).
It works by taking all .ulg files from the logs folder and then using
ulog2csv to writing the resulting .csv files in the tmp folder.
It finally writes all results in the plots folder.

## Usage
```
user@host: ls
logs tmp plots
user@host: python -m tulg.plot_main.py --bdir .
```
