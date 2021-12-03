from __future__ import print_function
from __future__ import unicode_literals

import argparse
import csv
import os
import shutil
import sys
import time
import numpy as np
import pyspark
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    OneHotEncoder,
    OneHotEncoderEstimator,
    StringIndexer,
    VectorAssembler,
    VectorIndexer,
)
from pyspark.sql.functions import *
from pyspark.sql.types import *


def csv_line(data):
    r = ",".join(str(d) for d in data[1])
    # print("CSV Data Neha")
    # print(data)
    # print(data[0])
    # print(data[1])
    return str(data[0]) + "," + r + "," + str(data[2]) + "," + str(data[3]) + "," + str(data[4])


def main():
    parser = argparse.ArgumentParser(description="app inputs and outputs")
    parser.add_argument("--s3_input_bucket", type=str, help="s3 input bucket")
    parser.add_argument("--s3_input_key_prefix", type=str, help="s3 input key prefix")
    parser.add_argument("--s3_output_bucket", type=str, help="s3 output bucket")
    parser.add_argument("--s3_output_key_prefix", type=str, help="s3 output key prefix")
    parser.add_argument("--s3_output_file_key", type=str, help="s3 output file key")
    args = parser.parse_args()

    spark = SparkSession.builder.appName("PySparkApp").getOrCreate()

    # This is needed to save RDDs which is the only way to write nested Dataframes into CSV format
    spark.sparkContext._jsc.hadoopConfiguration().set(
        "mapred.output.committer.class", "org.apache.hadoop.mapred.FileOutputCommitter"
    )

    # Defining the schema corresponding to the input data. The input data does not contain the headers
    schema = StructType(
        [
            StructField("App", StringType(), True),
            StructField("Category", StringType(), True),
            StructField("Rating", DoubleType(), True),
            StructField("Reviews", LongType(), True),
            StructField("Size", DoubleType(), True),
            StructField("Installs", LongType(), True),
            StructField("Type", StringType(), True),
            StructField("Price", DoubleType(), True),
            StructField("Content Rating", StringType(), True),
            StructField("Genres", StringType(), True)
        ]
    )

    # Downloading the data from S3 into a Dataframe
    total_df = spark.read.csv(
        ("s3://" + os.path.join("551grouproject6", args.s3_output_file_key)),
        header=False,
        schema=schema,
    )

    # split dataset
    (train_df, validation_df) = total_df.randomSplit([0.8, 0.2], seed=0)

    CONTI_FEATURES = ['Reviews', 'Size', 'Installs', 'Price']
    CATE_FEATURES = ['App', 'Category', 'Type', 'Content Rating', 'Genres']

    stages = []  # stages in our Pipeline
    for categoricalCol in CATE_FEATURES:
        stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol + "Index")
        #         encoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()],
        #                                          outputCols=[categoricalCol + "classVec"])
        stages += [stringIndexer]

    assemblerInputs = [c + "Index" for c in CATE_FEATURES] + CONTI_FEATURES

    assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
    stages += [assembler]

    # Create a Pipeline.
    pipeline = Pipeline(stages=stages)
    pipelineModel = pipeline.fit(train_df)
    train_df = pipelineModel.transform(train_df)
    validation_df = pipelineModel.transform(validation_df)

    # Convert the train dataframe to RDD to save in CSV format and upload to S3
    train_rdd = train_df.rdd.map(lambda x: (x.Rating, x.features, x.App, x.Category, x.Genres))
    # print("Train rdd neha")
    # print(train_rdd)
    train_lines = train_rdd.map(csv_line)
    # print("Train lines Neha")
    # print(train_lines)
    train_lines.saveAsTextFile(
        "s3://" + os.path.join(args.s3_output_bucket, args.s3_output_key_prefix, "train")
    )

    # Convert the validation dataframe to RDD to save in CSV format and upload to S3
    # validation_rdd = validation_df.rdd.map(lambda x: (x.Rating, x.features))
    # validation_lines = validation_rdd.map(csv_line)
    # validation_lines.saveAsTextFile(
    #    "s3://" + os.path.join(args.s3_output_bucket, args.s3_output_key_prefix, "validation")
    # )


if __name__ == "__main__":
    main()
