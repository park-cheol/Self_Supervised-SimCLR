ToDo
- Multi GPU 지원
- Resume, Valid set, Testset Code


# A Simple Framework for Contrastive Learning of Visual Representations

1. **Data Augmentation의 구성(Random Crop, Color Distortion)**
2. **Learnable Nonlinear Transformation**
3. **Larger Batch Sizes and More Training Steps**

![그래프](https://user-images.githubusercontent.com/76771847/157406654-8621247f-7782-47e3-8933-0f6e46068c39.png)

# Method

**# 전체 흐름**

![Screenshot from 2022-03-09 17-52-54](https://user-images.githubusercontent.com/76771847/157406649-b98c9777-7574-422a-a4e1-eb5b1fdbc2c3.png)

1. 동일한 이미지에 서로 다른 Augmentation 적용
2. Encoder(RESNET-50)을 이용해서 **Representation** 추출 
3. Reprensentation을 **Projection head**(2 MLP & RELU)을 통해 Embedding Space에 Mapping 
4. Positive와 Negative Examples 끼리 Sim(i.e Cosine Similarity)을 구한 후 Cross Entropy 계산 -> **NT-Xent**

**# Data Augmentation**

![Screenshot from 2022-03-09 18-00-03](https://user-images.githubusercontent.com/76771847/157407961-a01b05dd-070a-49d4-ac3a-2e0b66c74697.png)

- Random Crop과 Random Color Distortion을 같이 사용할 경우 가장 좋은 성능을 기록
- **Random Color Distortion** 없으면 Image Histogram이 비슷한 Distribution 형태를 가지게 되고 이는 신경망이 문제를 해결하기 위해 지름길을 이용하여 
**낮은 Representation Qulaity를 보여줌**
=> **Task의 난이도를 높히기 위해 Color Distortion 사용**
- 실험 결과 Color Distortion을 많이 가할수록 좋은 성능을 보여줌
  (Supervised 경우 반대로 더 안좋아짐)

**# Larger Batch Sizes & More Training Steps**

![Screenshot from 2022-03-09 18-11-40](https://user-images.githubusercontent.com/76771847/157409954-919aac8f-f8d8-405d-bcf6-8ffd229fd5a7.png)

- **Larger Batch Size(N)**: Negative Examples 2(N-1)을 더 많이 참고할 수 있음
- **More Training Steps**: Random Augmentation으로 많은 양의 Negative Examples가 있으므로 충분한 시간을 학습

**# Learnable Nonlinear Transformation**

![Screenshot from 2022-03-09 18-19-48](https://user-images.githubusercontent.com/76771847/157411396-05177334-ce08-473b-b59a-4b144b55ce74.png)

- Nonlinear Head가 있는 것이 항상 성능이 더 좋게 나옴(than linear / None Head)
- 하지만 Output Dimension은 성능에 큰 영향이 없어 보임

- **t-SNE (a): Encoder Output Vector , (b) Projection Head Output Vector**
![Screenshot from 2022-03-09 18-20-14](https://user-images.githubusercontent.com/76771847/157411405-6277922b-f91b-4b2c-96bc-95863e21f827.png)

- Projection Head가 성능을 향상 시켜주었지만 오히려 (a)가 더 잘 구분되는 것을 보여줌
- **Contrastive Loss에 의해 유도되어진 정보 손실이라고 추측**
- Projection Head는 Data transformation에 invariant 하도록 학습
- 그래서 projectiion Head는 DownStream에 유용할 수 있는 정보를 제거할 수 있음
- 결국 (a)에서 더 많은 정보들이 형성하고 유지할 수 있음.

# Reference

**paper: https://arxiv.org/abs/2002.05709**

**참고한 Code:**

**https://github.com/sthalles/SimCLR
https://github.com/Spijkervet/SimCLR**

# Usage
1. **Pretrain SimCLR**
> python main.py --gpu 0 --world-size 1 --dataset-name ...

2. **Linear Evaluation using pretrained SimCLR**
> python linear_evaluatin.py --gpu 0 --world-size 1 --dataset-name ... --resume (model.pth)
