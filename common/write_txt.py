import datetime
def write_list_data(name, data, time):
    name = "./data/" + name + "_data/train" + time + ".txt"
    f = open(name, "w")
    for line in data:
        f.write(str(line) + '\n')
    f.close()
    print("train list data write success")

def write_list_evaluate_data(name, data, time):
    name = "./data/" + name + "_data/evaluate" + time + ".txt"
    f = open(name, "w")
    for line in data:
        f.write(str(line) + '\n')
    f.close()
    print("evaluate list data write success")

def write_list_price_data(name, data, time):
    name = "./data/" + name + "_data/price" + time + ".txt"
    f = open(name, "w")
    for line in data:
        f.write(str(line) + '\n')
    f.close()
    print("price list data write success")

def write_list_social_data(name, data, time):
    name = "./data/" + name + "_data/social" + time + ".txt"
    f = open(name, "w")
    for line in data:
        f.write(str(line) + '\n')
    f.close()
    print("social data write success")

def write_dict_data(name, data, time):
    name = "./data/" + name + "_data/evaluate_result" + time + ".txt"
    f = open(name, "w")
    for item in data.items():
        f.write(str(item) + '\n')
    f.close()
    print("evaluate dict data write success")

# write_dict_data("qmix", {'reward': 666,
#                   "cpu_ratio": 66.66,
#                   "memory_ratio": 66.66,
#                   "price": 66666,
#                   "social_welfare": 666
#                 })