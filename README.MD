# Sea-ML

This repository offers machine learning models and tools designed to predict meteorological variables from coastal stations provided by MeteoGalicia, the regional meteorological service. It includes machine learning algorithms, primarily from the scikit-learn library, deployed through operational.py. The models are specifically trained on meteorological data collected from each station, with the directories named according to the respective station they represent.

## Table of Contents
- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
The **Sea-ML** project leverages machine learning techniques to process and analyze meteorological data from coastal stations. The data includes variables such as wind speed, temperature, humidity, and other weather-related parameters, which are used to train predictive models. The aim is to improve the forecasting of weather conditions in coastal areas, aiding regional environmental and operational planning.

## Directory Structure
Each directory corresponds to a coastal meteorological station from the MeteoGalicia service. These directories contain datasets, preprocessing scripts, and machine learning models tailored for specific stations.

```bash
sea-ml/
│
├── coron/            # Data and models for the Corón station
├── marin/            # Data and models for the Marín station
├── ons/              # Data and models for the Ons station
└── udra/             # Data and models for the Udra station
```

