import time, sys

import datetime

time_to_start = datetime.datetime.strptime(sys.argv[1], "%H:%M:%S,%f")

time_to_start += datetime.timedelta(0,90)

time_to_start = time_to_start - datetime.datetime.strptime(datetime.datetime.utcnow().strftime("%H:%M:%S,%f"), "%H:%M:%S,%f") 

#time.sleep(time_to_start)

print(time_to_start.total_seconds())
time_to_start = time_to_start.total_seconds()
while(time_to_start):
    time.sleep(1)
    time_to_start -= 1
    print(time_to_start)




