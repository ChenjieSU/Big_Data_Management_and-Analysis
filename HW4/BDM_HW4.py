from pyspark import SparkContext
import csv
import pyproj
import shapely.geometry as geom
from heapq import nlargest
import heapq
import sys


def createBoroughsIndex(shapefile):
    
    import rtree
    import fiona.crs
    import geopandas as gpd
    zones1 = gpd.read_file(shapefile).to_crs(fiona.crs.from_epsg(2263))
    index1 = rtree.Rtree()
    for idx,geometry in enumerate(zones1.geometry):
        index1.insert(idx, geometry.bounds)
    return (index1, zones1)

def createNeighborhoodsIndex(shapefile):

    import rtree
    import fiona.crs
    import geopandas as gpd
    zones2 = gpd.read_file(shapefile).to_crs(fiona.crs.from_epsg(2263))
    index2 = rtree.Rtree()
    for idx,geometry in enumerate(zones2.geometry):
        index2.insert(idx, geometry.bounds)
    return (index2, zones2)

def findPickUpZone(p, index1, zones1):
    
    match1 = index1.intersection((p.x, p.y, p.x, p.y))
    for idx in match1:
        if zones1.geometry[idx].contains(p):
            return zones1.boroname[idx]
    return None

def findDropZone(q, index2, zones2):
    
    match2 = index2.intersection((q.x, q.y, q.x, q.y))
    for idx in match2:
        if zones2.geometry[idx].contains(q):
            return zones2.neighborhood[idx]
    return None

def processTrips(pid, records):
    
    import csv
    import pyproj
    import shapely.geometry as geom
    
    # Create an R-tree index
    proj = pyproj.Proj(init="epsg:2263", preserve_units=True)    
    index1, zones1 = createBoroughsIndex('boroughs.geojson')   
    index2, zones2 = createNeighborhoodsIndex('neighborhoods.geojson')   

    
    # Skip the header
    if pid==0:
        next(records)
    reader = csv.reader(records)
    counts = {}
    output={}
    
    for row in reader:
        
        try:
            p = geom.Point(proj(float(row[5]), float(row[6]))) 
            q = geom.Point(proj(float(row[9]), float(row[10])))
            match1 = None
            match2 = None
            b_zone = findPickUpZone(p, index1, zones1)
            n_zone = findDropZone(q, index2, zones2)
            nb_zone = b_zone,n_zone
            
            if nb_zone:
                counts[nb_zone] = counts.get(nb_zone, 0) + 1 
        
        except(ValueError, IndexError):
            pass
                
          
 
    return counts.items()

def toCSVLine(data):
    return ','.join(str(d) for d in data)

if __name__== "__main__":

    sc=SparkContext()

    sys_input = sys.argv[1]
    sys_output = sys.argv[2]
    
    rdd = sc.textFile(sys_input)
    rdd.mapPartitionsWithIndex(processTrips) \
        .reduceByKey(lambda x,y: x+y) \
        .filter(lambda x: x[0][0] != None) \
        .filter(lambda x: x[0][1] != None) \
        .map(lambda x: (x[0][0],x[0][1],x[1]))  \
        .groupBy(lambda x:x[0]) \
        .flatMap(lambda g: nlargest(3,g[1],key=lambda x:x[2])) \
        .map(lambda x:(x[0],(x[1],x[2]))) \
        .groupByKey().mapValues(list) \
        .sortByKey() \
        .map(toCSVLine) \
        .saveAsTextFile(sys_output)
                 
