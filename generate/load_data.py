import csv

def load_data(data_size):
    # 从CSV文件中读取数据并存入列表
    data_list = []
    if data_size == 0:
        with open('small_output.csv', 'r') as csvfile:
            csv_reader = csv.DictReader(csvfile)
            for row in csv_reader:
                data_list.append(dict(row))
    if data_size == 1:
        with open('output.csv', 'r') as csvfile:
            csv_reader = csv.DictReader(csvfile)
            for row in csv_reader:
                data_list.append(dict(row))
    if data_size == 2:
        with open('big_output.csv', 'r') as csvfile:
            csv_reader = csv.DictReader(csvfile)
            for row in csv_reader:
                data_list.append(dict(row))

    print("{} data load success".format(len(data_list)))
    return data_list
    # # 打印结果
    # for data in data_list:
    #     print(data)
