# Sea-ML

This repository contains machine learning models for forecasting and analyzing meteorological data from coastal meteorological stations in the region of Galicia, provided by the MeteoGalicia regional meteorological service. The models and data are aimed at enhancing weather prediction and climate analysis for coastal areas.

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
├── cecebre/            # Data and models for the Cecebre station
├── coruna/             # Data and models for the A Coruña station
├── ferrol/             # Data and models for the Ferrol station
├── marin/              # Data and models for the Marín station
├── muxia/              # Data and models for the Muxía station
└── vigo/               # Data and models for the Vigo station
