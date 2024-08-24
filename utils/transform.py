import numpy.typing as npt

def transform_time(time):
    """
        transform the time of seconds to {} hour {} miniute {} second
    """
    hour, time = divmod(time, 60 * 60)
    minute, time = divmod(time, 60)
    second = time
    return hour, minute, second

class DimensionTransformer:
    pass
