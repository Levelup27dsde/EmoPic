![example](images/example.png)



 EmoPic is combined by two words, **emo**tion + **pic**ture. EmoPic is a service which make changes in facial expressions through Korean text sentiment analysis. The sentiment analysis of Korean text is done by KoBert. The StarGan model is used for making facial expressions changes in pictures.

 Getting two inputs, text from utterance and a portrait photo, the EmoPic service analyzes the emotion implied in the input text and photoshops the input picture changing the face expression to the analyzed sentiment.

 Text sentiment analysis model is trained my utterance text and face expression changing model is trained by selfies of Koreans with various categories.

KoBert : https://github.com/SKTBrain/KoBERT

StarGan : https://github.com/yunjey/stargan

Team Levelup : [Kyeongwon Cho](https://github.com/F1RERED), Yukyeong Kang, Taeyoung Kim, Hyoeun Ahn

--------------

## Demonstration videos

<Data dashboard & Login & Signup>

<video src="videos/data_dashboard,login,sign_up.mp4"></video>

<Direct inputs & slang [Surprised]>

<video src="videos/[surprised]slang,direct_input.mp4"></video>

<Using image2text(Naver Cloud OCR) API & screenshotlayer API [Sad]>

<video src="videos/[Sad]img2txt_API,screenshoturl_API.mp4"></video>

<Distribution server(gunicorn & nginx)  [Angry])>

<video src="videos/[angry]distributed_server.mp4"></video>

--------------------------------

## Files to be replaced

```
EmoPic

 └ mysite

	 └ .cache

		 └  kobert_v1.zip

	└ mysite

		└ surprise.pt


	└ stargan

		└ models

			└ 300000-D.ckpt

			└ 300000-G.ckpt

```

* kobert_v1.zip

  should be the trained model of kobert

+ surprise.pt

  should be the trained model of sentiment analysis based on kobert

+ 300000-D.ckpt / 300000-G.ckpt 

  should be the trained model of stargan 

-------------------------

## Requirements

 Simulation based on python 3.7

 Versions in [requirements.txt](requirements.txt) are not mandatory. It's only the version used for our project.

 Please refer to the KoBert's and the StarGan's dependecies.

---------------------------------

## etc.

![certificate](images/[D27]_상장(문제해결빅데이터활용프로젝트)_최우수상_3조.jpg)