#!/usr/bin/env python

from __future__ import print_function

"""Dump binary log generated by PX4's sdlog2 or APM as CSV
    
Usage: python sdlog2_dump.py <log.bin> [-v] [-e] [-d delimiter] [-n null] [-m MSG[_field1,field2,...]]
Example Usage: python sdlog2_dump.py log.bin -m GPS_Lat,Lng -m AHR2_Roll,Pitch,Yaw
    
    -v  Use plain debug output instead of CSV.
    
	-e	Recover from errors.
    
    -d  Use "delimiter" in CSV. Default is ",".
    
    -n  Use "null" as placeholder for empty values in CSV. Default is empty.
    
    -m MSG[_field1,field2,...]
        Dump only messages of specified type, and only specified fields.
        Multiple -m options allowed.
	"""

__author__  = "Anton Babushkin"
__version__ = "1.2"

import struct, sys

if sys.hexversion >= 0x030000F0:
    runningPython3 = True
    def _parseCString(cstr):
        return str(cstr, 'ascii').split('\0')[0]
else:
    runningPython3 = False
    def _parseCString(cstr):
        return str(cstr).split('\0')[0]

class SDLog2Parser:
    BLOCK_SIZE = 8192
    MSG_HEADER_LEN = 3
    MSG_HEAD1 = 0xA3
    MSG_HEAD2 = 0x95
    MSG_FORMAT_PACKET_LEN = 89
    MSG_FORMAT_STRUCT = "BB4s16s64s"
    MSG_TYPE_FORMAT = 0x80
    FORMAT_TO_STRUCT = {
        "b": ("b", None),
        "B": ("B", None),
        "h": ("h", None),
        "H": ("H", None),
        "i": ("i", None),
        "I": ("I", None),
        "f": ("f", None),
        "d": ("d", None),
        "n": ("4s", None),
        "N": ("16s", None),
        "Z": ("64s", None),
        "c": ("h", 0.01),
        "C": ("H", 0.01),
        "e": ("i", 0.01),
        "E": ("I", 0.01),
        "L": ("i", 0.0000001),
        "M": ("b", None),
        "q": ("q", None),
        "Q": ("Q", None),
    }
    __csv_delim = ","
    __csv_null = ""
    __msg_filter = []
    __time_msg = None
    __debug_out = False
    __correct_errors = False
    __file_name = None
    __file = None
    
    def __init__(self):
        return
    
    def reset(self):
        self.__msg_descrs = {}      # message descriptions by message type map
        self.__msg_labels = {}      # message labels by message name map
        self.__msg_names = []       # message names in the same order as FORMAT messages
        self.__buffer = bytearray() # buffer for input binary data
        self.__ptr = 0              # read pointer in buffer
        self.__csv_columns = []     # CSV file columns in correct order in format "MSG.label"
        self.__csv_data = {}        # current values for all columns
        self.__csv_updated = False
        self.__msg_filter_map = {}  # filter in form of map, with '*" expanded to full list of fields
    
    def setCSVDelimiter(self, csv_delim):
        self.__csv_delim = csv_delim
    
    def setCSVNull(self, csv_null):
        self.__csv_null = csv_null
    
    def setMsgFilter(self, msg_filter):
        self.__msg_filter = msg_filter
    
    def setTimeMsg(self, time_msg):
        self.__time_msg = time_msg
    
    def setDebugOut(self, debug_out):
        self.__debug_out = debug_out

    def setCorrectErrors(self, correct_errors):
        self.__correct_errors = correct_errors

    def setFileName(self, file_name):
    	self.__file_name = file_name
    	if file_name != None:
    		self.__file = open(file_name, 'w+')
    	else:
    		self.__file = None

    
    def process(self, fn):
        self.reset()
        if self.__debug_out:
            # init __msg_filter_map
            for msg_name, show_fields in self.__msg_filter:
                self.__msg_filter_map[msg_name] = show_fields
        first_data_msg = True
        f = open(fn, "rb")
        bytes_read = 0
        while True:
            chunk = f.read(self.BLOCK_SIZE)
            if len(chunk) == 0:
                break
            self.__buffer = self.__buffer[self.__ptr:] + chunk
            self.__ptr = 0
            while self.__bytesLeft() >= self.MSG_HEADER_LEN:
                head1 = self.__buffer[self.__ptr]
                head2 = self.__buffer[self.__ptr+1]
                if (head1 != self.MSG_HEAD1 or head2 != self.MSG_HEAD2):
                    if self.__correct_errors:
                        self.__ptr += 1
                        continue
                    else:
                        raise Exception("Invalid header at %i (0x%X): %02X %02X, must be %02X %02X" % (bytes_read + self.__ptr, bytes_read + self.__ptr, head1, head2, self.MSG_HEAD1, self.MSG_HEAD2))
                msg_type = self.__buffer[self.__ptr+2]
                if msg_type == self.MSG_TYPE_FORMAT:
                    # parse FORMAT message
                    if self.__bytesLeft() < self.MSG_FORMAT_PACKET_LEN:
                        break
                    self.__parseMsgDescr()
                else:
                    # parse data message
                    msg_descr = self.__msg_descrs.get(msg_type, None)
                    if msg_descr == None:
                        if self.__correct_errors:
                            self.__ptr += 1
                            continue
                        else:
                            raise Exception("Unknown msg type: %i" % msg_type)
                    msg_length = msg_descr[0]
                    if self.__bytesLeft() < msg_length:
                        break
                    if first_data_msg:
                        # build CSV columns and init data map
                        if not self.__debug_out:
                            self.__initCSV()
                        first_data_msg = False
                    self.__parseMsg(msg_descr)
            bytes_read += self.__ptr
            if not self.__debug_out and self.__time_msg != None and self.__csv_updated:
                self.__printCSVRow()
        f.close()
    
    def __bytesLeft(self):
        return len(self.__buffer) - self.__ptr
    
    def __filterMsg(self, msg_name):
        show_fields = "*"
        if len(self.__msg_filter_map) > 0:
            show_fields = self.__msg_filter_map.get(msg_name)
        return show_fields
    
    def __initCSV(self):
        if len(self.__msg_filter) == 0:
            for msg_name in self.__msg_names:
                self.__msg_filter.append((msg_name, "*"))
        for msg_name, show_fields in self.__msg_filter:
            if show_fields == "*":
                show_fields = self.__msg_labels.get(msg_name, [])
            self.__msg_filter_map[msg_name] = show_fields
            for field in show_fields:
                full_label = msg_name + "_" + field
                self.__csv_columns.append(full_label)
                self.__csv_data[full_label] = None
        if self.__file != None:
            print(self.__csv_delim.join(self.__csv_columns), file=self.__file)
        else:
            print(self.__csv_delim.join(self.__csv_columns))

    def __printCSVRow(self):
        s = []
        for full_label in self.__csv_columns:
            v = self.__csv_data[full_label]
            if v == None:
                v = self.__csv_null
            else:
                v = str(v)
            s.append(v)

        if self.__file != None:
            print(self.__csv_delim.join(s), file=self.__file)
        else:
            print(self.__csv_delim.join(s))

    def __parseMsgDescr(self):
        if runningPython3:
            data = struct.unpack(self.MSG_FORMAT_STRUCT, self.__buffer[self.__ptr + 3 : self.__ptr + self.MSG_FORMAT_PACKET_LEN])
        else:
            data = struct.unpack(self.MSG_FORMAT_STRUCT, str(self.__buffer[self.__ptr + 3 : self.__ptr + self.MSG_FORMAT_PACKET_LEN]))
        msg_type = data[0]
        if msg_type != self.MSG_TYPE_FORMAT:
            msg_length = data[1]
            msg_name = _parseCString(data[2])
            msg_format = _parseCString(data[3])
            msg_labels = _parseCString(data[4]).split(",")
            # Convert msg_format to struct.unpack format string
            msg_struct = ""
            msg_mults = []
            for c in msg_format:
                try:
                    f = self.FORMAT_TO_STRUCT[c]
                    msg_struct += f[0]
                    msg_mults.append(f[1])
                except KeyError as e:
                    raise Exception("Unsupported format char: %s in message %s (%i)" % (c, msg_name, msg_type))
            msg_struct = "<" + msg_struct   # force little-endian
            self.__msg_descrs[msg_type] = (msg_length, msg_name, msg_format, msg_labels, msg_struct, msg_mults)
            self.__msg_labels[msg_name] = msg_labels
            self.__msg_names.append(msg_name)
            if self.__debug_out:
                if self.__filterMsg(msg_name) != None:
                    print("MSG FORMAT: type = %i, length = %i, name = %s, format = %s, labels = %s, struct = %s, mults = %s" % (
                                msg_type, msg_length, msg_name, msg_format, str(msg_labels), msg_struct, msg_mults))
        self.__ptr += self.MSG_FORMAT_PACKET_LEN
    
    def __parseMsg(self, msg_descr):
        msg_length, msg_name, msg_format, msg_labels, msg_struct, msg_mults = msg_descr
        if not self.__debug_out and self.__time_msg != None and msg_name == self.__time_msg and self.__csv_updated:
            self.__printCSVRow()
            self.__csv_updated = False
        show_fields = self.__filterMsg(msg_name)
        if (show_fields != None):
            if runningPython3:
                data = list(struct.unpack(msg_struct, self.__buffer[self.__ptr+self.MSG_HEADER_LEN:self.__ptr+msg_length]))
            else:
                data = list(struct.unpack(msg_struct, str(self.__buffer[self.__ptr+self.MSG_HEADER_LEN:self.__ptr+msg_length])))
            for i in range(len(data)):
                if type(data[i]) is str:
                    data[i] = _parseCString(data[i])
                m = msg_mults[i]
                if m != None:
                    data[i] = data[i] * m
            if self.__debug_out:
                s = []
                for i in range(len(data)):
                    label = msg_labels[i]
                    if show_fields == "*" or label in show_fields:
                        s.append(label + "=" + str(data[i]))
                print("MSG %s: %s" % (msg_name, ", ".join(s)))
            else:
                # update CSV data buffer
                for i in range(len(data)):
                    label = msg_labels[i]
                    if label in show_fields:
                        self.__csv_data[msg_name + "_" + label] = data[i]
                        if self.__time_msg != None and msg_name != self.__time_msg:
                            self.__csv_updated = True
                if self.__time_msg == None:
                    self.__printCSVRow()
        self.__ptr += msg_length


def _main():
    if len(sys.argv) < 2:
        print("Usage: python sdlog2_dump.py <log.bin> [-v] [-e] [-d delimiter] [-n null] [-m MSG[_field1,field2,...]] [-t TIME_MSG_NAME]\n")
        print("\t-v\tUse plain debug output instead of CSV.\n")
        print("\t-e\tRecover from errors.\n")
        print("\t-d\tUse \"delimiter\" in CSV. Default is \",\".\n")
        print("\t-n\tUse \"null\" as placeholder for empty values in CSV. Default is empty.\n")
        print("\t-m MSG[_field1,field2,...]\n\t\tDump only messages of specified type, and only specified fields.\n\t\tMultiple -m options allowed.")
        print("\t-t\tSpecify TIME message name to group data messages by time and significantly reduce duplicate output.\n")
        print("\t-fPrint to file instead of stdout")
        return
    fn = sys.argv[1]
    debug_out = False
    correct_errors = False
    msg_filter = []
    csv_null = ""
    csv_delim = ","
    time_msg = "TIME"
    file_name = None
    opt = None
    for arg in sys.argv[2:]:
        if opt != None:
            if opt == "d":
                csv_delim = arg
            elif opt == "n":
                csv_null = arg
            elif opt == "t":
                time_msg = arg
            elif opt == "f":
            	file_name = arg
            elif opt == "m":
                show_fields = "*"
                a = arg.split("_")
                if len(a) > 1:
                    show_fields = a[1].split(",")
                msg_filter.append((a[0], show_fields))
            opt = None
        else:
            if arg == "-v":
                debug_out = True
            elif arg == "-e":
                correct_errors = True
            elif arg == "-d":
                opt = "d"
            elif arg == "-n":
                opt = "n"
            elif arg == "-m":
                opt = "m"
            elif arg == "-t":
                opt = "t"
            elif arg == "-f":
                opt = "f"

    if csv_delim == "\\t":
        csv_delim = "\t"
    parser = SDLog2Parser()
    parser.setCSVDelimiter(csv_delim)
    parser.setCSVNull(csv_null)
    parser.setMsgFilter(msg_filter)
    parser.setTimeMsg(time_msg)
    parser.setFileName(file_name)
    parser.setDebugOut(debug_out)
    parser.setCorrectErrors(correct_errors)
    parser.process(fn)


if __name__ == "__main__":
    _main()
