# innstall java
!apt-get install openjdk-8-jdk-headless -qq > /dev/null

# install spark (change the version number if needed)
!wget -q https://archive.apache.org/dist/spark/spark-3.0.0/spark-3.0.0-bin-hadoop3.2.tgz

# unzip the spark file to the current folder
!tar xf spark-3.0.0-bin-hadoop3.2.tgz

# set your spark folder to your system path environment.
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] = "/content/spark-3.0.0-bin-hadoop3.2"


# install findspark using pip
!pip install -q findspark
!pip install pyspark

import findspark
findspark.init()
findspark.find()

import pyspark
from pyspark import SparkConf
from pyspark.sql import SparkSession

spark=SparkSession\
      .builder\
      .appName("ReadandWrite")\
      .getOrCreate()

spark

df = spark.read.csv("amenity.csv", header=True, inferSchema=True)
df.printSchema()

df.show(5)

#unique value in column amenity
df.select("amenity").distinct().show()

#chek data type
df.dtypes

df.columns

"""analyze column longitude-latitude"""

df.select('longitude-lattitude').show()

"""Drop first column because it is not meaningful"""

df=df.drop('_c0')
df.show()

"""split longitude and lattitude into new column and remove special char so that values in column longitude and lattitude will be the number only"""

#pyspark.sql.functions.split(str, pattern, limit=-1)
from pyspark.sql.functions import split, col, substring, length, expr

df=df.withColumn('longitude', split(df['longitude-lattitude'], '^\W|,').getItem(1)) \
  .withColumn('lattitude', split(df['longitude-lattitude'], ',|\W$').getItem(1))

df.show(truncate=False)

df.dtypes

"""1. convert dtype of column longitude and latitude to float
2. drop column longitude-lattitude
"""

#Drop column longitude-lattitude
df=df.drop('longitude-lattitude')

#Convert dtype
newdf = df.selectExpr("cast(name as string) name",
    "cast(amenity as string) amenity",
    "cast(All_tags as string) All_tags",
    "cast(longitude as float) longitude",
    "cast(lattitude as float) lattitude")

newdf.printSchema()
newdf.show(truncate=False)

"""Drop column *All_tags* because the values are mix and we dont need the values for the analysis"""

newdf2=newdf.drop('All_tags')
newdf2.show(3)

"""# Check for missing values for each column"""

#check for missing values
from pyspark.sql import functions as F

newdf2.select([F.count(F.when(F.isnull(c), c)).alias(c) for c in newdf2.columns]).show()

"""Check for number of rows before removing missing values"""

newdf2.count()

"""Removing missing values"""

newdf2=newdf2.dropna()

"""Number of rows after removing missing values"""

newdf2.count()

"""Again, we will check if there is any other missing values"""

newdf2.select([F.count(F.when(F.isnull(c), c)).alias(c) for c in newdf2.columns]).show()

"""# Check and removing duplicate rows

1. Count distinct rows and compare with total rows
"""

distinctDF = newdf2.distinct()
print("Distinct count: "+str(distinctDF.count()))

"""Total rows are 53893 but distinct rows are only 53882 thus there are some duplicates rows. So we will remove the duplicate rows so that the number of rows will be the same as distinct rows

2. Drop duplicate rows
"""

newdf2 = newdf2.dropDuplicates()
print("Distinct count: "+str(newdf2.count()))

"""Now the data has no missing values, has been cleaned from unnecessary columns and duplicate rows. Next, we will analyze the data using EDA"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def barplot(df, col, lim=None, yname=None):

    '''
    This function makes a bar plot for the Spark dataframe df
    for categorical column col only selecting top categories
    as specified by lim.
    '''

    # Grouping by the categories, counting on each categories
    # and ordering them by the count
    classes = df.groupBy(col).count().orderBy('count', ascending=False)

    # Take first 'lim' number of rows and convert to pandas
    pd_df = classes.limit(lim).toPandas()

    # Making plot
    plt.rcParams["figure.figsize"] = [10, 3.50]
    pd_df.plot(kind='bar', x=col, legend=False)
    plt.ylabel(yname)
    plt.show()

barplot(newdf2, 'amenity', lim=5, yname='Total')

barplot(newdf2, 'amenity', lim= 75, yname='Total')

newdf2.collect()[0]['longitude']

newdf2.select('longitude').collect()

newdf2.select('longitude','lattitude').collect()[0]

"""# Plot the longitude and latitude"""

!pip install geopandas

import geopandas as gpd
from shapely.geometry import Point

geometry = [Point(xy) for xy in zip( newdf2.select('longitude').collect(), newdf2.select('lattitude').collect())]
gdf = gpd.GeoDataFrame(newdf2.collect(), geometry=geometry)
gdf.plot(cmap='GnBu', figsize=(10,10))          #cmap=gradient color

"""# Identify 5 locations based on longitude and lattitude"""

!pip install geopy

from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent="geoapiExercises")

newdf2.collect()[0]['lattitude']

a=0
while (a<=4):
  location = geolocator.reverse(newdf2.select('lattitude','longitude').collect()[a])
  a+=1
  print(location)
