'''
Various useful functions and other that do not necessarily fit in
other places.
'''

import datetime


def date_time_str(microseconds=False):
    '''
    Return the current date and time as a string from current year to
    current second or microsecond, 
    for example 20240118123904 for 18.01.2024 at 12.39:04.
    '''

    fstring = "%Y%m%d%H%M%S"
    if microseconds:
        fstring += "%f"

    return datetime.datetime.now().strftime(fstring)