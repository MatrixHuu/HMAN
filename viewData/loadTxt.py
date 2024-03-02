import pickle

# 打开文本文件
# with open('../data/MSVD/youtube_mapping.txt', 'r') as f:
#     # 按行读取文件内容
#     lines = f.readlines()
#
# # 创建一个空字典用于存储键值对
# dict= {}
#
# # 处理每一行内容
# for line in lines:
#     line = line.strip()  # 去除每行末尾的换行符和空格
#     if line:
#         vname, vid = line.split(' ')
#         dict[vname] = str(int(vid[3:]) - 1)
#
# with open('vidname_list.pkl', 'wb') as f:
#     pickle.dump(dict, f)

file_path = '/home/valca3090/Mark/HMAN/viewData/vidname_list.pkl'

# 使用pickle模块加载.pkl文件
with open(file_path, 'rb') as file:
    name_to_id = pickle.load(file)

print("end")
