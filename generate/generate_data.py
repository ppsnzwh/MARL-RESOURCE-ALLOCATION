import csv
import random
import math
import random

from decimal import *


def generate_data(num):
    # 生成数据
    data = []
    for i in range(num):
        if i % 10 == 0 and i != 0:
            # small data
            cpu = random.randint(1, 8)
            memory = random.randint(1, 8)
            demand_time = max(math.ceil(cpu / 4), math.ceil(memory / 4))
            r = round(random.uniform(0.9, 1.1), 2)
            bid = round(float(Decimal(str(2.5 * cpu + 1.5 * memory))) * float(Decimal(str(r))), 2)
            # bid = (2.5 * cpu + 1.5 * memory) * round(random.uniform(0.9, 1.1), 2)
            row = [i, cpu, memory, demand_time, -1, -1, -1, bid, -1]
            data.append(row)
            continue
        elif i % 15 == 0 and i != 0:
            # big data
            cpu = random.randint(1, 50)
            memory = random.randint(1, 50)
            demand_time = max(math.ceil(cpu / 4), math.ceil(memory / 4))
            r = round(random.uniform(0.9, 1.1), 2)
            bid = round(float(Decimal(str(2.5 * cpu + 1.5 * memory))) * float(Decimal(str(r))), 2)
            # bid = (2.5 * cpu + 1.5 * memory) * round(random.uniform(0.9, 1.1), 2)
            row = [i, cpu, memory, demand_time, -1, -1, -1, bid, -1]
            data.append(row)
            continue
        else:
            # normal data
            cpu = random.randint(11, 16)
            memory = random.randint(11, 16)
            demand_time = max(math.ceil(cpu / 4), math.ceil(memory / 4))
            r = round(random.uniform(0.9, 1.1), 2)
            bid = round(float(Decimal(str(2.5 * cpu + 1.5 * memory))) * float(Decimal(str(r))), 2)
            # bid = (2.5 * cpu + 1.5 * memory) * round(random.uniform(0.9, 1.1), 2)
            row = [i, cpu, memory, demand_time, -1, -1, -1, bid, -1]
            data.append(row)
            continue



    # 写入CSV文件
    header = ['id', 'cpu', 'memory', 'demand_time', 'start_time', 'end_time', 'is_run', 'bid', 'reserve']
    with open('output.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(header)
        csv_writer.writerows(data)
    print("normal data generate success")

def generate_big_data(num):
    # 生成数据
    data = []
    for i in range(num):
        if i % 10 == 0 and i != 0:
            # small data
            cpu = random.randint(16, 32)
            memory = random.randint(16, 32)
            demand_time = max(math.ceil(cpu / 4), math.ceil(memory / 4))
            r = round(random.uniform(0.9, 1.1), 2)
            bid = round(float(Decimal(str(2.5 * cpu + 1.5 * memory))) * float(Decimal(str(r))), 2)
            # bid = (2.5 * cpu + 1.5 * memory) * round(random.uniform(0.9, 1.1), 2)
            row = [i, cpu, memory, demand_time, -1, -1, -1, bid, -1]
            data.append(row)
            continue
        else:
            # big data
            cpu = random.randint(16, 50)
            memory = random.randint(16, 50)
            demand_time = max(math.ceil(cpu / 4), math.ceil(memory / 4))
            r = round(random.uniform(0.9, 1.1), 2)
            bid = round(float(Decimal(str(2.5 * cpu + 1.5 * memory))) * float(Decimal(str(r))), 2)
            # bid = (2.5 * cpu + 1.5 * memory) * round(random.uniform(0.9, 1.1), 2)
            row = [i, cpu, memory, demand_time, -1, -1, -1, bid, -1]
            data.append(row)
            continue
    # 写入CSV文件
    header = ['id', 'cpu', 'memory', 'demand_time', 'start_time', 'end_time', 'is_run', 'bid', 'reserve']
    with open('big_output.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(header)
        csv_writer.writerows(data)
    print("big data generate success")

def generate_small_data(num):
    # 生成数据
    data = []
    for i in range(num):
        if i % 10 == 0 and i != 0:
            # small data
            cpu = random.randint(1, 4)
            memory = random.randint(1, 8)
            demand_time = max(math.ceil(cpu / 4), math.ceil(memory / 4))
            r = round(random.uniform(0.9, 1.1), 2)
            bid = round(float(Decimal(str(2.5 * cpu + 1.5 * memory))) * float(Decimal(str(r))), 2)
            # bid = (2.5 * cpu + 1.5 * memory) * round(random.uniform(0.9, 1.1), 2)
            row = [i, cpu, memory, demand_time, -1, -1, -1, bid, -1]
            data.append(row)
            continue
        else:
            # normal data
            cpu = random.randint(1, 8)
            memory = random.randint(11, 16)
            demand_time = max(math.ceil(cpu / 4), math.ceil(memory / 4))
            r = round(random.uniform(0.9, 1.1), 2)
            bid = round(float(Decimal(str(2.5 * cpu + 1.5 * memory))) * float(Decimal(str(r))), 2)
            # bid = (2.5 * cpu + 1.5 * memory) * round(random.uniform(0.9, 1.1), 2)
            row = [i, cpu, memory, demand_time, -1, -1, -1, bid, -1]
            data.append(row)
            continue
    # 写入CSV文件
    header = ['id', 'cpu', 'memory', 'demand_time', 'start_time', 'end_time', 'is_run', 'bid', 'reserve']
    with open('small_output.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(header)
        csv_writer.writerows(data)
    print("small data generate success")

# generate_data(500)
# generate_small_data(500)
# generate_big_data(500)

