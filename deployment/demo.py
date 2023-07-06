from classification import classify
path = r'C:\Users\time3\Desktop\SupernovaDL\yolov8\test'
print(classify(path, True)) #示例程序

filename = 'bogus.jpg'
print(classify(filename, False))
