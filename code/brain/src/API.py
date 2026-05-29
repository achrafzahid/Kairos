from iex_cppparser import parse_dates
from iex_cppparser import compile_cpp

if __name__ == "__main__" :
# Download and parse data over a date   range
    parse_dates(
        start_date="2024-07-05", 
        end_date="2024-07-05", 
        download_dir="./pcap", 
        parsed_folder="./parsed", 
        symbol="symbols.txt", 
        download=True, 
        # split=True # CRITICAL for laptops: Prevents memory crashes
    )