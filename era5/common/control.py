import os
from typing import List, Set
from collections import Counter
import datetime as dt


class DataQualityControl:

    def __init__(self, dataroot: str):
        self.dataroot: str = dataroot
        self.filenames: List[str] = sorted([
            fname for fname in os.listdir(path=self.dataroot) if fname.endswith('.grib')
        ])

    def check_missing_files(self, fromdate: str, todate: str) -> List[str]:
        fromdate: dt.date = dt.datetime.strptime(fromdate, '%Y%m%d')
        todate: dt.date = dt.datetime.strptime(todate, '%Y%m%d')
        downloaded_files: Set[str] = set(self.filenames)
        required_files: Set[str] = {
            (fromdate + dt.timedelta(days=x)).strftime('%Y%m%d') + '.grib' 
            for x in range((todate - fromdate).days + 1)
        }
        return sorted(required_files - downloaded_files)

    def check_incompleted_files(self) -> List[str]:
        filebytes: List[int] = [os.stat(f'{self.dataroot}/{fname}').st_size for fname in self.filenames]
        expected_file_byte: int = Counter(filebytes).most_common(n=1).pop()[0]
        return [self.filenames[i] for i in range(len(self.filenames)) if filebytes[i] != expected_file_byte]
    

if __name__ == '__main__':
    dataroot: str = 'data/2d/era5/wind/1000'
    self = DataQualityControl(dataroot)
    print(f"Missing files in {dataroot}:\n{self.check_missing_files(fromdate='20170101', todate='20240731')}")
    print('-' * 10)
    print(f"Incomplete files in {dataroot}:\n{self.check_incompleted_files()}")

