import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format
from pyspark.sql.types import *
from pyspark.sql.functions import monotonically_increasing_id


config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID']=config['AWS']['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY']=config['AWS']['AWS_SECRET_ACCESS_KEY']


def create_spark_session():
    """Gets an existing SparkSession or, if there is no existing one, creates a new one
    
    Returns
    ----------
    spark: SparkSession
        A spark session object 
    """
    
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.5") \
        .getOrCreate()
    return spark


def process_song_data(spark):
    """Reads song data files from S3, creates songs and artists tables, and load them to a new S3 bucket
    
    Parameters
    ----------
    spark: SparkSession
        The spark session object we created
    """
    
    # get filepath to song data file
    song_data = "s3a://udacity-dend/song_data/A/A/A/*.json"
    
    # read song data file
    df = spark.read.json(song_data)

    # extract columns to create songs table
    songs_table = df[['song_id', 'title', 'artist_id', col('year').cast(IntegerType()), col('duration').cast(DoubleType())]] \
                    .distinct()
    
    # write songs table to parquet files partitioned by year and artist_id
    songs_table.write.partitionBy("year", "artist_id").parquet("s3a://sparkify-tables/songs.parquet")

    # extract columns to create artists table
    artists_table = df[['artist_id', 'artist_name', 'artist_location', \
                        col('artist_latitude').cast(DoubleType()), col('artist_longitude').cast(DoubleType())]] \
                        .distinct() \
                        .withColumnRenamed('artist_name', 'name') \
                        .withColumnRenamed('artist_location', 'location') \
                        .withColumnRenamed('artist_latitude', 'latitude') \
                        .withColumnRenamed('artist_longitude', 'longitude')
    
    # write artists table to parquet files
    artists_table.write.parquet("s3a://sparkify-tables/artists.parquet")


def process_log_data(spark):
    """Reads log data files from S3, creates users, time and songplays tables, and load them to a new S3 bucket
    
    Parameters
    ----------
    spark: SparkSession
        The spark session object we created
    """
    
    # get filepath to log data file
    log_data = "s3a://udacity-dend/log_data/2018/11/*.json"

    # read log data file
    df = spark.read.json(log_data)
    
    # filter by actions for song plays
    df = df.filter("page == 'NextSong'")

    # extract columns for users table    
    users_table = df[[col('userId').cast(IntegerType()), 'firstName', 'lastName', 'gender', 'level']] \
                    .distinct() \
                    .withColumnRenamed('userId', 'user_id') \
                    .withColumnRenamed('firstName', 'first_name') \
                    .withColumnRenamed('lastName', 'last_name')
    
    # write users table to parquet files
    users_table.write.parquet("s3a://sparkify-tables/users.parquet")

    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda x: datetime.fromtimestamp(x / 1000.0), TimestampType())
    df = df.withColumn("timestamp", get_timestamp(df['ts']))
    
    # create datetime column from original timestamp column
    get_datetime = udf(lambda x: datetime.fromtimestamp(x / 1000.0), DateType())
    df = df.withColumn("datetime", get_datetime(df['ts']))
    
    # extract columns to create time table
    time_table = df[['timestamp', hour(col('timestamp')), dayofmonth(col('timestamp')), weekofyear(col('timestamp')), \
                 month(col('timestamp')), year(col('timestamp')), date_format(col('timestamp'), 'EEE')]] \
                    .distinct() \
                    .withColumnRenamed('hour(timestamp)', 'hour') \
                    .withColumnRenamed('dayofmonth(timestamp)', 'day') \
                    .withColumnRenamed('weekofyear(timestamp)', 'week') \
                    .withColumnRenamed('month(timestamp)', 'month') \
                    .withColumnRenamed('year(timestamp)', 'year') \
                    .withColumnRenamed('date_format(timestamp, EEE)', 'weekday')
    
    # write time table to parquet files partitioned by year and month
    time_table.write.partitionBy("year", "month").parquet("s3a://sparkify-tables/time.parquet")

    # read in song data to use for songplays table
    song_df = spark.read.json("s3a://udacity-dend/song_data/A/A/A/*.json")

    # extract columns from joined song and log datasets to create songplays table 
    songplays_table = song_df.join(df, (song_df['artist_name'] == df['artist']) & (song_df['title'] == df['song'])) \
                    [[col('timestamp').alias('start_time'), col('userId').alias('user_id').cast(IntegerType()), \
                      'level', 'song_id', 'artist_id', col('sessionId').alias('session_id').cast(IntegerType()), \
                      'location', col('userAgent').alias('user_agent')]] \
                    .withColumn('songplay_id', monotonically_increasing_id())

    # write songplays table to parquet files partitioned by year and month
    funcColname = [(year, "year"), (month, "month")]
    exprs = [col('*')] + [func(col('start_time')).alias(colname) for func, colname in funcColname]
    songplays_table.select(exprs).write.partitionBy("year", "month").parquet("s3a://sparkify-tables/songplays.parquet")


def main():
    """
    Gets or creates a spark session, then calls process_song_data and process_log_data to start the ETL
    """
    spark = create_spark_session()
    
    process_song_data(spark)    
    process_log_data(spark)


if __name__ == "__main__":
    main()
