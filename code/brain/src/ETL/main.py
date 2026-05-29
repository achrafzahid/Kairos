from extract import extract_day, randotendays
from transform import build_and_save_deeplob_tensors, return_csv_path,cleanday
from load import load_tensor,remove_csv, remove_pcap
from datetime import datetime
from iex_cppparser import compile_cpp
# the idea is want to build a parser that saves some limit order book  of each second of the day for a certain asset
# it should download the data if its non existing use the parser, transform it, save it into a directory of the day with
# each directory having the day as the name and the csv numbered with a timestamp in the current day
# and after that we delete the file of the day



# for the selection we will select the period of time July 2024 – September 2024
# you may ask why :
#           This is arguably the best recent 3-month window to pick right now. It contains a perfect, compact cycle of regimes.
#           The Calm/Uptrend: July 2024 was characterized by a steady, low-volatility climb in the S&P 500 (SPY).
#           The Extreme Volatility: Early August 2024 (specifically around August 5th). The sudden unwinding of the Japanese Yen carry trade
#           caused a massive, historic spike in the VIX. The limit order books during those few days were incredibly thin and chaotic—perfect
#           for training your model on stress conditions.
#           The Recovery Trend: Late August through September featured a strong, directional recovery rally.


YEAR = 2024

if __name__ == "__main__" :
    dates = []
    # important note =========
    # keep in mind the days that are off
    for i in range(7,10) :
        ran = randotendays()
        for r in ran :
            dates.append(str(datetime(year=YEAR,month=i, day=r))[:10])
    dates = ["2024-07-05",
            "2024-07-09",
            "2024-07-11",
            "2024-07-12",
            "2024-07-16",
            "2024-07-17",
            "2024-07-18",
            "2024-07-23",
            "2024-07-24",
            "2024-07-25",
            "2024-08-02",
            "2024-08-05",
            "2024-08-06",
            "2024-08-09",
            "2024-08-12",
            "2024-08-20",
            "2024-08-21",
            "2024-08-26",
            "2024-08-27",
            "2024-08-29",
            "2024-09-10",
            "2024-09-12",
            "2024-09-13",
            "2024-09-18",
            "2024-09-20",
            "2024-09-23",
            "2024-09-24",
            "2024-09-25",
            "2024-09-27",
            "2024-09-30"]    
    for i in range(len(dates)) :
        d = dates[i]
        if return_csv_path(d) == None :
            extract_day(d)
        csv_path = return_csv_path(d)
        dic = cleanday(csv_path)
        for key, df in dic.items() :
            tensor = build_and_save_deeplob_tensors(ticker=key,df=df,date_str=d)
            load_tensor(normalized_tensor=tensor,ticker=key, date_str=d)
        dic.clear()
        # remove_csv(csv_path=csv_path)
        remove_pcap()
        
        
        
        