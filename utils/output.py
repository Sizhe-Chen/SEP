import time
import datetime

def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

class LogProcessBar():
    def __init__(self, logfile, args):
        self.last_time = time.time()
        self.begin_time = time.time()
        self.logfile = logfile
        with open(self.logfile, 'a') as f:
            f.write(str(args) + '\n')

    def log(self, msg):
        with open(self.logfile, 'a') as f:
            f.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\t' + msg + '\n')

    def refresh(self, current, total, mode, msg=None):
        if current == 0:
            self.begin_time = time.time()  # Reset for new bar.

        cur_time = time.time()
        step_time = cur_time - self.last_time
        self.last_time = cur_time
        tot_time = cur_time - self.begin_time

        L = []
        L.append("[{:>3d}/{:<3d}]".format(current + 1, total))
        L.append(" {} |".format(mode.center(6)))
        L.append(' Step:{}'.format(format_time(step_time).ljust(6)))
        L.append('| Tot:{}'.format(format_time(tot_time).ljust(8)))

        if msg:
            L.append(' | ' + msg)
        msg = ''.join(L)
        if current < total - 1:
            print('\r', msg, end='')
        elif current == total - 1:
            print('\r', msg)
            with open(self.logfile, 'a') as f:
                f.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")+'\t'+msg+'\n')
        else:
            raise NotImplementedError



last_time = time.time()
begin_time = time.time()

def progress_bar( current, total, mode, msg=None):
    global last_time,begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append("[{:>3d}/{:<3d}]".format(current + 1, total))
    L.append(" {} |".format(mode.center(6)))
    L.append(' Step:{}'.format(format_time(step_time).ljust(6)))
    L.append('| Tot:{}'.format(format_time(tot_time).ljust(8)))

    if msg:
        L.append(' | ' + msg)
    msg = ''.join(L)
    if current < total - 1:
        print('\r', msg, end='')
    elif current == total - 1:
        print('\r', msg)




