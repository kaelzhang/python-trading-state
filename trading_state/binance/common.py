from datetime import datetime


def timestamp_to_datetime(timestamp: int) -> datetime:
    """
    Convert a timestamp in milliseconds to a datetime object
    """
    return datetime.fromtimestamp(timestamp / 1000)
