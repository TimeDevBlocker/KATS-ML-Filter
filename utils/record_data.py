import csv
from classification import classify
def write_list_of_dicts_to_csv(data_list, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # 写入表头
        writer.writerow(['file_name', 'bogus_conf', 'true_conf'])
        
        # 写入数据
        for item in data_list:
            file_name = item.get('file_name', '')
            bogus_conf = item.get('bogus_conf', '')
            true_conf = item.get('true_conf', '')
            writer.writerow([file_name, bogus_conf, true_conf])

path = r'C:\Users\time3\Desktop\SupernovaDL\yolov8\datasets\kats\val\bogus'
data_list = classify(path, True) #示例程序
# 调用函数写入CSV文件
write_list_of_dicts_to_csv(data_list, 'output.csv')
