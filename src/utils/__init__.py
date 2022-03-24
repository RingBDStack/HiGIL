from datetime import datetime


def get_current_datetime():
    return datetime.now().strftime('%Y_%m_%d_%H:%M:%S')
