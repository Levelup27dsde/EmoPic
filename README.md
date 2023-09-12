![example](example.png)



 EmoPic is service which make changes in facial expressions through Korean text sentiment analysis. The sentiment analysis of Korean text is done by KoBert. The StarGan model is used for making facial expressions changes in pictures.

KoBert : https://github.com/SKTBrain/KoBERT

StarGan : https://github.com/yunjey/stargan

Team : Kyeongwon Cho, Yukyeong Kang, Taeyoung Kim, Hyoeun Ahn



--------------------------------

## Files to be replaced
```
EmoPic

 └ mysite

​	 └ .cache

​		 └  kobert_v1.zip

​	└ mysite

​		└ surprise.pt

​	└ stargan

​		└ models

​			└ 300000-D.ckpt

​			└ 300000-G.ckpt
```
* kobert_v1.zip

  ​	should be the trained model of kobert

+ surprise.pt

  ​	should be the trained model of sentiment analysis based on kobert

+ 300000-D.ckpt / 300000-G.ckpt 

  ​	should be the trained model of stargan 

-------------------------

## Requirements

 Simulation based on python 3.7
 
 Versions in requirements.txt are not mandatory. It's only the version used for our project.

 Please refer to the KoBert's and the StarGan's dependecies.
