#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from pyspark import SparkContext
from pyspark.sql.session import SparkSession
import csv
import numpy as np
import time 
import sys


start_time = time.time()


def processViolationCounty(pid, records):
    # Mahattan->1, Bronx->2, Brooklyn->3, Queens->4, Staten Island ->5
    county_idx = {'MAN':1,'MH':1,'MN':1,'NEWY':1,'NEW Y':1,'NY':1,\
                  'BRONX':2,'BX':2,'PBX':2,\
                  'BK':3,'K':3,'KING':3,'KINGS':3,\
                  'Q':4,'QN':4,'QNS':4,'QU':4,'QUEEN':4,\
                  'R':5,'RICHMOND':5}
    
    # Skip the header
    if pid==0:
        next(records)
    reader = csv.reader(records)
    
    for row in reader:
        try:
            year = int(row[4][-4:])
            street = row[24].lower()
            
            if row[21] in county_idx.keys():
                boro = county_idx[row[21]]
            else:
                continue
                
            if row[23].isdigit():
                is_left = int(row[23]) % 2
                house = float(row[23])
                
            else:
                try:
                    first, house = row[23].split('-')
                    house = str(int(house))
                    is_left = int(house) % 2
                    house = float(first+'.'+house)
                        
                except:
                    continue
                    
        except:
            continue
            
        yield (year, house, street, boro, is_left)
        
def processCenterLine(pid,records):
    
     boro_idx = {'1': 'Manhattan', '2': 'Bronx', '3': 'Brooklyn', '4':'Queens','5': 'Staten Island'} 
        
    if pid==0:
        next(records)
    reader = csv.reader(records)
    
    for row in reader:
        physical_id = int(row[0])
        street = row[28].lower()
        boro = boro_idx[row[13]]
        
        for i in [2, 3, 4, 5]:
            if row[i]:
                if row[i].isdigit():
                    row[i] = float(row[i]) 
                else:
                    first, row[i] = row[i].split('-')
                    row[i] = str(int(row[i]))
                    row[i] = float(first+'.'+row[i])   
            else:
                    row[i] = 0.0
        
        yield (physical_id, street, boro, row[2], row[3], 1)
        yield (physical_id, street, boro, row[4], row[5], 0)
        

def processFormat(records):
    for r in records:
        if r[0][1]==2015:
            yield (r[0][0], (r[1], 0, 0, 0, 0))
        elif r[0][1]==2016:
            yield (r[0][0], (0, r[1], 0, 0, 0))
        elif r[0][1]==2017:
            yield (r[0][0], (0, 0, r[1], 0, 0))
        elif r[0][1]==2018:
            yield (r[0][0], (0, 0, 0, r[1], 0))
        elif r[0][1]==2019:
            yield (r[0][0], (0, 0, 0, 0, r[1]))
        else: 
            yield (r[0][0], (0, 0, 0, 0, 0))

def compute_ols(y, x=list(range(2015,2020))):
    x, y = np.array(x), np.array(y)
    # number of observations
    n = np.size(x)
    # mean of x, y
    m_x, m_y = np.mean(x), np.mean(y)
    # cross-deviation and deviation of x
    xy = np.sum(y*x) - n*m_y*m_x 
    xx = np.sum(x*x) - n*m_x*m_x
    # regression coefficients 
    coef = xy / xx
    return float(str(coef))

if __name__ == "__main__":
    output = sys.argv[1]
    
    sc = SparkContext()
    spark = SparkSession(sc)

    centerline = sc.textFile('hdfs:///tmp/bdm/nyc_cscl.csv')
    rdd_cl = centerline.mapPartitionsWithIndex(processCenterLine)
    cl_df = spark.createDataFrame(rdd_cl,('physical_id','street','boro','low','high','is_left'))
    
    violations = sc.textFile('hdfs:///tmp/bdm/nyc_parking_violation/')
    rdd_vio = violations.mapPartitionsWithIndex(processViolationCounty)
    vio_df = spark.createDataFrame(rdd_vio,('year','house','street','boro','is_left'))
    
    boro_condition = (vio_df.boro == cl_df.boro)
    street_condition = (vio_df.street == cl_df.street)
    is_left_condition = (vio_df.is_left == cl_df.is_left)
    house_condition = (vio_df.house >= cl_df.low)&(vio_df.house <= cl_df.high)
    
    condition = [boro_condition,street_condition,is_left_condition,house_condition]

    df = cl_df.join(vio_df,condition,'left').groupby([cl_df.physical_id, vio_df.year]).count()

    df = df.fillna(0)

    df.rdd.map(lambda x: ((x[0], x[1]), x[2])) \
            .mapPartitions(processFormat) \
            .reduceByKey(lambda x,y: (x[0]+y[0], x[1]+y[1], x[2]+y[2], x[3]+y[3], x[4]+y[4])) \
            .sortByKey() \
            .mapValues(lambda y: y + (compute_ols(y=list(y)),)) \
            .map(lambda x: ((x[0],) + x[1])) \
            .map(lambda x: (str(x)[1:-1])) \
            .saveAsTextFile(output)    
    
    print('time taken:', time.time() - start_time)

    
    
