import numpy as np
from museek.time_ordered_data import TimeOrderedData

def flag_percent_recv(data: TimeOrderedData):
    """
    return the flag percent for each receiver
    """
    flag_percent = []
    receivers_list = []
    for i_receiver, receiver in enumerate(data.receivers):
        flag_recv = data.flags.get(recv=i_receiver)
        flag_recv_combine = flag_recv.combine(threshold=1)
        flag_percent.append(round(np.sum(flag_recv_combine.array>=1)/len(flag_recv_combine.array.flatten()), 4))
        receivers_list.append(str(receiver))

    return receivers_list, flag_percent
