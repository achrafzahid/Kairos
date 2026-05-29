from iex_cppparser import parse_dates
from datetime import datetime
import random as rd
# Download and parse data over a date range
# parse_dates(
#     start_date="2017-08-29", 
#     end_date="2017-08-29", 
#     download_dir="./pcap", 
#     parsed_folder="./parsed", 
#     symbol="symbols.txt", 
#     download=True, 
#     split=True # CRITICAL for laptops: Prevents memory crashes
# )
rd.seed(1337)




def randotendays() :
    s = set()
    while(len(s) != 10) :
        s.add(rd.uniform(1,30).__floor__())
    return sorted(list(s))


def extract_day(date : str) :
    parse_dates(
        start_date=str(date), 
        end_date=str(date), 
        download_dir="../pcap", 
        parsed_folder="../parsed", 
        symbol="../symbols.txt", 
        download=True, 
    )


