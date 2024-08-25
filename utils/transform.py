import time

def cal_time(descrip_state: str):
    def decorator(func):
        def decorate(*args, **kwargs):
            start = time.time()
            func(*args, **kwargs)
            end = time.time()
            hour, minute, second = transform_time(end - start)
            print('-'*120)
            print('{}，用时 {:.2f} 时 {:.2f} 分 {:.2f} 秒'.format(descrip_state, hour, minute,  second))
            print('-'*120)
        return decorate
    return decorator

def transform_time(time):
    """
        transform the time of seconds to {} hour {} miniute {} second
    """
    hour, time = divmod(time, 60 * 60)
    minute, time = divmod(time, 60)
    second = time
    return hour, minute, second
