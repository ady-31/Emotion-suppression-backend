# Emotion Suppression Detection API

A FastAPI-based REST API for detecting emotional suppression by analyzing facial Action Units (AU) and valence data from video analysis.

## 🎯 Overview

This API processes facial expression data to compute **emotion suppression scores**. It analyzes the relationship between facial muscle activity (Action Units) and emotional valence to identify moments when emotions may be suppressed.

## 🚀 Features

- **Multiple File Upload**: Analyze multiple AU and valence CSV file pairs simultaneously
- **Per-File Analysis**: Get individual results for each file pair
- **Overall Statistics**: Combined metrics across all processed files
- **Interactive API Docs**: Built-in Swagger UI for easy testing

## 📋 Requirements

- Python 3.8+
- FastAPI
- Uvicorn
- Pandas
- NumPy

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/Emotion-suppression-api.git
   cd Emotion-suppression-api
   ```

2. **Install dependencies**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

3. **Run the server**
   ```bash
   uvicorn main:app --reload
   ```

4. **Open the API docs**
   ```
   http://127.0.0.1:8000/docs
   ```

## 📡 API Endpoints

### `GET /`
Health check endpoint.

**Response:**
```json
{
  "status": "API is running"
}
```

### `POST /analyze`
Analyze emotion suppression from AU and valence CSV files.

**Parameters:**
- `au_files`: One or more AU CSV files (multipart/form-data)
- `valence_files`: One or more valence CSV files (multipart/form-data)

**Response:**
```json
{
  "overall": {
    "mean_suppression": 0.4123,
    "std_suppression": 0.0912,
    "total_frames": 1824,
    "files_processed": 2
  },
  "per_file": [
    {
      "file_pair": 1,
      "mean_suppression": 0.41,
      "std_suppression": 0.09,
      "frames": 912
    },
    {
      "file_pair": 2,
      "mean_suppression": 0.42,
      "std_suppression": 0.10,
      "frames": 912
    }
  ]
}
```

## 📁 Input File Format

### AU CSV File
CSV file with columns starting with "AU" (e.g., AU01, AU02, AU04, etc.):
```csv
frame,AU01,AU02,AU04,AU05,...
1,0.5,0.3,0.1,0.0,...
2,0.6,0.4,0.2,0.1,...
```

### Valence CSV File
CSV file with a numeric column for valence values:
```csv
frame,valence
1,0.65
2,0.72
```

## 🔬 How It Works

1. **Load Data**: Reads AU and valence CSV files
2. **Compute AU Energy**: Sums all Action Unit intensities per frame
3. **Normalize**: Scales AU energy to [0, 1] range
4. **Compute Suppression**: Multiplies AU energy by the absolute change in valence
5. **Temporal Smoothing**: Applies a 3-frame moving average filter
6. **Aggregate Results**: Returns mean, std, and frame count statistics

## 📂 Project Structure

```
Emotion-suppression-api/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── requirements.txt     # Python dependencies
│   └── suppression/
│       ├── __init__.py
│       └── logic.py         # Core suppression algorithms
└── README.md
```

## 📝 License

MIT License

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
