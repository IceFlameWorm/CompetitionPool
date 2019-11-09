# 基于OCR的身份证要素提取

- [比赛首页](https://www.datafountain.cn/competitions/346)
- [数据集]()

## baselines

1. [baseline1](baselines/id_card_detection/baseline1): 通过图像增强、轮廓检测等方式检测出（bounding box）身份证在图像中位置。
2. [baseline2](baselines/id_card_detection/baseline2): 通过模板匹配，透视变换等方式，从原始图像中检测并抽取出身份证。