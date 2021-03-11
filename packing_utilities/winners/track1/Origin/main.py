import os
import sys
import json
import time
import mypack
import itertools
import numpy as np
from multiprocessing.pool import Pool
from order import Order
from pack import Pack
from routing import Routing 



def main(argv):
    input_dir = argv[1]
    output_dir = argv[2]
    for file_name in os.listdir(input_dir):
        print("The order ", file_name, " is processing...")
        input_path = os.path.join(input_dir, file_name)
        with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
            message_str = f.read()
        order = Order(message_str)
        mypack_obj = mypack.Pack(order)
        hwpack_obj = Pack(message_str, mypack_obj=mypack_obj)
        route = Routing(order, pack_obj=hwpack_obj)
        nondominated_sol_listlist = route.population_global_search2(1400, 7)
        for sol_list in nondominated_sol_listlist:
            route.res["solutionArray"].append(route.sol_2_trucklist_Dict[''.join([str(x) for x in sol_list])])
        route.save_sols(output_dir)
        print("The order ", file_name, " is done.")


if __name__ == "__main__":
    main(sys.argv)
