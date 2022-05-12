# Signful
Submission for Hack The North 2021 demo linkL: https://devpost.com/software/signful

<img width="673" alt="Screen Shot 2022-05-11 at 10 37 46 PM" src="https://user-images.githubusercontent.com/67188030/167980990-6a9b3d7d-b6ea-4dde-beb4-b2d920c925b8.png">


# Inspiration
There are 70 million hearing-impaired people all over the world, of which 6 million people are in US/Canada.

Of these 6 million people, only 600,000 people know American Sign Language. This is an extremely low percentage!

One reason for this is that many hearing impaired people don't bother to learn sign language is because if they actually learn it, most people around them don't know sign language to understand.

This is a real pain point that doesn't have any viable solutions yet currently. That's why we decided to build Signful !!!

# Pain point
From the computer vision perspective, there are two kinds of actions in sign language: single-frame actions and multi-frame actions.

Recognizing single-frame actions are actually very easy, if not the most basic task that can be achieved by the likes of OpenCV or a basic neural network. For example: "I Love You"

![signful1](https://user-images.githubusercontent.com/67188030/167980740-146ea008-39df-44bb-adb5-d01257738f11.jpeg)

But what about multi-frame actions?

Currently, there is NO tech in the world whatsoever that can identity multi-frame actions in sign language, even the most common expressions like “Hello” or “Thank you”.

![signful2](https://user-images.githubusercontent.com/67188030/167980766-afb6c4cf-0e18-4ece-bca3-6f3f39e6fa8e.gif)
![signful3](https://user-images.githubusercontent.com/67188030/167980775-8b72d047-27c2-414b-8759-af82a57d2e20.gif)


## What it does
Signful allows users to record themselves then uses Computer Vision and Machine Learning to translating their American Sign Language into English texts. AND it is the first one that can identify both single-frame and multi-frame actions in sign language.

## How we built it
Unlike traditional image recoginition techniques which use pixels, we used Google Mediapipe API and OpenCV to auto detect and apply keypoints into face, hand and body frames.

We collected these keypoints and processed them by putting them into numpy arrays.

We inputted these arrays into our own custom-built Neural Network on Tensorflow. Tensorflow trained and generated the model weights.

Finally we built an API on top of our trained model using Python and Flask

## Challenges we ran into
This is our first time to work with Computer Vision and OpenCV and a motion tracking tech like Mediapipe so there's a bit of learning curve here.

The Machine Learning model was also not easy to fine-tune the parameters.

We have troubles to send live video frames back to the server. We tried to use Socket.io and then HTTP requests but found out Flask server could not handle too many frames per second sent.

## Accomplishments that we're proud of
We are able to make a MVP to detect 3 phrases: 'Hello', 'Thanks', 'I Love You'. These are phrases based on sequences so it is not possible to do it with simple image classification technique, we are very proud to pull this off !!!

## What we learned
We learned a lot about Computer Vision, Machine Learning, Websocket and Video streaming.

## What's next for Signful
We will further train the model with more words and improve the overall precision and speed

