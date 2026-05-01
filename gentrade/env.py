"""Module-level config. Boto3 is loaded lazily so import doesn't require AWS credentials."""

import os

HIT = "TARGET_HIT"
STOPPED = "STOPPED_OUT"
NA = "NO_CLOSE_IN_WINDOW"
NO_TRADES = "NO_ENTRIES_FOR_STRATEGY"

# risk:reward ratio 2:1
TARGET = 0.015
STOP_LOSS = TARGET / 2

MAX_GENERATIONS = 100
MIN_INDICATORS = int(os.getenv("MIN_INDICATORS", 2))
MAX_SAME_CLASS_INDICATORS = int(os.getenv("MAX_SAME_CLASS_INDICATORS", 2))
MAX_STRATEGY_INDICATORS = int(os.getenv("MAX_STRATEGY_INDICATORS", 4))
POPULATION_SIZE = int(os.getenv("POPULATION_SIZE", 10))
CONJUNCTIONS = ["and", "or", "and not", "or not"]
CUTOFF_PERCENT = 1 / 3
BUY_AMOUNT = 1000

AWS_REGION = os.getenv("AWS_REGION", "eu-west-1")


def get_s3_resource():
    import boto3

    return boto3.resource("s3", region_name=AWS_REGION)
