TRAIN = 'Train'
VALID = 'Valid'
TEST = 'Test'
UNLABELED = 'Unlabeled'
NO_LABEL = ''
ALL = 'ALL'
CORRECT = 'Correct'
WRONG = 'Wrong'
UNKNOWN = 'Unknown'


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    v_str = str(v).lower().strip()
    if v_str in ('true', 'yes', 't', 'y', '1', 'on'):
        return True
    if v_str in ('false', 'no', 'f', 'n', '0', 'off', 'none', 'null', ''):
        return False
    return False