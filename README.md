# Tempo-and-Key-Estimation using CNN
## MMT Final Project
- #### [Sai Akarsh](https://github.com/saiakarsh193) (2019111017)
- #### [Nikit Uppala](https://github.com/Nikit-Uppala) (2019101022)

## Goal
We are implementing axis-aligned directional CNNs for estimating tempo and key. This is done by calculating the CQT spectrogram of the audio for key estimation and running it through the CNN and similarly calculating the Mel spectrogram of the audio for tempo estimation and running it through the CNN for training.

Report of our project can be found [here](https://docs.google.com/presentation/d/10bv1G8l7KESA6rWir5OEQgTZ4UDsTwwfXVSWDUQTrC0/edit?usp=sharing)

## Datasets chosen
- [Extended Ballroom](http://anasynth.ircam.fr/home/media/ExtendedBallroom) for Tempo Estimation Training
- [Ballroom](http://mtg.upf.edu/ismir2004/contest/tempoContest/node5.html) for Tempo Estimation Testing
- [Gainsteps](https://github.com/GiantSteps/giantsteps-key-dataset) for Key Estimation

## Results

|         Model         | Accuracy (%)  |
|:---------------------:|:-------------:|
|     shallow_tempo     |     82.80     |
|      shallow_key      |     42.14     |
| shallow_tempo_control |     49.57     |
|  shallow_key_control  |     07.53     |
|       deep_tempo      |     85.53     |
|        deep_key       |     48.10     |

## References
- Link to [paper](https://arxiv.org/pdf/1903.10839.pdf)
- Paper implementation GitHub [repository](https://github.com/hendriks73/directional_cnns)