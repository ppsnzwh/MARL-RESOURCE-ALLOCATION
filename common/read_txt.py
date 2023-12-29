
def read_list():
    list = []
    with open("../data/qmix_no_share_rnn_data/train1213222310.txt", "r", encoding='utf-8') as f:  # 打开文本
        for ann in f.readlines():
            data = float(ann.strip('\n'))  # 读取文本
            list.append(data)

    return list


def read_dict():
    with open("../data/qmix_data/evaluate_result1212143225.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
    dict_data = {}
    for line in lines:
        value = line.strip('\n').strip('()').strip("'").strip("'").split(",")
        value[0] = value[0].strip("'")
        dict_data[value[0]] = float(value[1])

