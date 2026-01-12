import os
import kagglehub

# 设置自定义缓存路径
os.environ['KAGGLEHUB_CACHE'] = '/Users/jorahmormont/PycharmProjects/BigDataFinalProject'

# 下载数据集
path = kagglehub.dataset_download("robikscube/flight-delay-dataset-20182022")
print("Path to dataset files:", path)