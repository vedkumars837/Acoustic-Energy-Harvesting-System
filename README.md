# Acoustic-Energy-Harvesting-System
Real-time Streamlit dashboard for acoustic energy harvesting simulation. Monitors sound intensity, energy accumulation, capacitor charging (0-100%), and IoT packet transmission. Features live waveforms, interactive Plotly gauges, silence detection &amp;amp; countdown timers. Python-based 📊⚡
# 🔊 Acoustic Energy Harvesting IoT System
### Team Byte Brigade - INNOVIT 2026

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)

---

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Installation & Setup](#installation--setup)
- [Usage Guide](#usage-guide)
---

## 🎯 Project Overview

**Acoustic Energy Harvesting IoT System** converts ambient noise pollution into usable electrical energy for powering low-power IoT devices. This project addresses India's dual challenge of noise pollution and battery-dependent IoT infrastructure.

### Key Highlights:
- 🔋 **100% Battery-Free** - No battery replacement needed
- ♻️ **Sustainable** - Converts noise into power
- 📡 **IoT Ready** - Powers wireless sensors
- 🇮🇳 **Atmanirbhar Bharat** - Made in India
- 🎯 **Smart Cities** - Railway, highway deployments

---

## ⚠️ Problem Statement

Urban India generates continuous high-decibel noise (80-110 dB) that remains **unutilized as energy**. Railway stations, highways, and industrial zones have massive untapped acoustic potential.

### Key Statistics:
- Railway platforms: **90-110 dB** during trains
- Highway traffic: **75-85 dB** continuous
- Industrial zones: **85-105 dB**
- Major stations: **2-3 million daily footfall**

---

## ✨ Features

### Software Dashboard:
- ✅ Real-time waveform visualization
- ✅ Energy metrics (µW/m², J, RMS)
- ✅ IoT simulation (capacitor, packets)
- ✅ Interactive Plotly gauges
- ✅ Silence detection
- ✅ Auto-transmission at 80%
- ✅ Countdown timer

---

## 🛠️ Technology Stack

**Core:**
- Python 3.8+, Streamlit 1.28+

**Audio:**
- PyAudio (44.1kHz, 256-sample chunks)
- NumPy (signal processing, FFT)

**Visualization:**
- Matplotlib (waveforms, bars)
- Plotly (interactive gauges)

---

## 📦 Installation & Setup

### Prerequisites:
```bash
python --version  # 3.8+
pip --version
```

### Quick Start:
```bash
# Clone repository
git clone https://github.com/ByteBrigade/acoustic-energy-harvesting.git
cd acoustic-energy-harvesting

# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run acoustix.py
```

### Platform-Specific PyAudio:

**Windows:**
```bash
pip install pyaudio
```

**macOS:**
```bash
brew install portaudio
pip install pyaudio
```

**Linux:**
```bash
sudo apt-get install python3-pyaudio
```

---

## 📖 Usage Guide

### Starting:
1. Run: `streamlit run acoustix.py`
2. Click **"🎙️ Start"** in sidebar
3. Make sound to see energy harvesting

### Dashboard Sections:
- **Live Metrics** - Intensity, energy, RMS, status
- **Waveform** - Real-time vibration plot
- **Charts** - Intensity & energy bars
- **IoT Display** - Capacitor %, packets, countdown
- **Gauges** - Interactive capacitor & packet meters

### Configuration:
- **Microphone Gain:** 1.0-10.0 (default: 3.0)
- **Wave Gain:** 1.0-300.0 (default: 150)
- **Energy Boost:** 1.0-1000.0 (default: 500)
- **Capacitor Max:** 0.1-10.0 J (default: 1.0 J)
- **Packet Cost:** 0.00001-0.001 J (default: 0.0001 J)
- **Auto-Transmit:** 50-100% (default: 80%)

---

<div align="center">

### 🌟 Star this repository if you find it useful!

### Ideas Powering Atmanirbhar Bharat 🇮🇳

**Made by Team Byte Brigade**

</div>