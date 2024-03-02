import pickle

# 打开.pkl文件并加载数据
with open('vidname_list.pkl', 'rb') as f:
    data = pickle.load(f)

# 打印数据
print(data)
