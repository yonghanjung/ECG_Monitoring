# -*- coding: utf-8 -*-
'''
Goal : CSV Reader
Author : Yonghan Jung, ISyE, KAIST 
Date : 150511
Comment 
- 

'''

''' Library '''
import csv
''' Function or Class '''


class HansCSV:
    def __init__(self):
        pass


    def Reader(self, FileName, Delimiter, Header):
        with open(FileName, 'rb') as csvfile:
            CSVReader = csv.reader(csvfile, delimiter = Delimiter)
            if Header:
                Header = CSVReader.next()
                Column = dict()
                for h in Header:
                    Column[h] = []
                for row in CSVReader:
                    for h, v in zip(Header, row):
                        Column[h].append(v)
            else:
                Column = []
                for row in CSVReader:
                    Row = []
                    Row.append(row.split(Delimiter))
                    Column.append(Row)

        return Column

    def Writer(self, FileName, MyDictList):
        FieldName = MyDictList[0].keys()
        HansFile = open(FileName, 'wb')
        wtr = csv.DictWriter(HansFile, delimiter='|', fieldnames=FieldName )
        for idx,row in enumerate(MyDictList):
            try:
                wtr.writerow(row)
            except:
                print "ERROR in {IDX}".fotmat(IDX = idx)

        HansFile.close()
        return None



if __name__ == "__main__":
    print None