from datetime import datetime


def get_timestamp():
    """Returns current timestamp to append to files"""
    current_timestamp = datetime.now()
    processed_timestamp = (
        str(current_timestamp)[:-7].replace(" ", "_").replace(":", "").replace("-", "") + "_"
    )

    return processed_timestamp
