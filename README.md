# 🌱 Organic Farm Pest Management AI

An AI-powered pest detection and organic treatment recommendation system that helps farmers identify crop pests and provides sustainable, environmentally-friendly treatment solutions.

## 🚀 Features

- **AI Pest Detection**: Upload images to identify harmful pests using deep learning
- **Organic Treatment Recommendations**: Get comprehensive organic pest management advice
- **Economic Impact Assessment**: Understand the potential damage and treatment urgency
- **Multi-language Support**: Available in multiple languages
- **Real-time Analysis**: Instant pest identification and treatment suggestions

## 📋 Prerequisites

Before running the application, you need to install:

1. **Python 3.8+**
2. **Ollama** (for AI-powered recommendations)

## 🔧 Ollama Installation & Setup

### Step 1: Download and Install Ollama

1. **Visit the official Ollama website**: [https://ollama.ai](https://ollama.ai)
2. **Download Ollama** for Windows
3. **Run the installer** and follow the installation wizard
4. **Restart your computer** after installation

### Step 2: Verify Ollama Installation

Open PowerShell and verify Ollama is installed:

```powershell
ollama --version
```

### Step 3: Pull the Required Model

The application uses the `gemma3n:latest` model. Pull it using PowerShell:

```powershell
# Pull the main model (this will take some time - ~7.5GB download)
ollama pull gemma3n:latest

# Verify the model is installed
ollama list
```

**Note**: The model download is approximately 7.5GB and may take 10-30 minutes depending on your internet connection.

### Step 4: Test the Model

Test that the model is working correctly:

```powershell
ollama run gemma3n:latest "Hello, can you help with organic farming?"
```

You should see a response from the AI model. Press `Ctrl+D` or type `/bye` to exit the chat.

## 🛠️ Application Setup

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd oip_pest
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify Project Structure

Ensure you have the following structure:
```
oip_pest/
├── app.py
├── config/
│   ├── config.py
│   └── languages.py
├── models/
│   └── class_names.json
├── presentation/
│   ├── ollama_service.py
│   ├── pest_interface.py
│   └── pest_predictor.py
└── requirements.txt
```

## 🚀 Running the Application

### 1. Start Ollama Service

Ensure Ollama is running (it usually starts automatically after installation):

```powershell
# Check if Ollama is running
ollama list

# If not running, start it
ollama serve
```

### 2. Launch the Application

```bash
python app.py
```

### 3. Access the Web Interface

Open your browser and navigate to:
```
http://localhost:7860
```

## 📱 How to Use

1. **Upload an Image**: Click the upload area and select a pest image
2. **Get Detection Results**: The AI will identify the pest and show confidence scores
3. **View Treatment Recommendations**: Receive detailed organic treatment advice
4. **Follow the Guidelines**: Implement the suggested organic pest management strategies

## 🔍 Supported Pest Classes

The system can detect and provide treatment recommendations for various harmful pests including:
- Aphids
- Armyworms
- Beetles
- Caterpillars
- And more...

## ⚙️ Configuration

### Model Configuration

The application is configured to use `gemma3n:latest` by default. You can modify the model in `config/config.py`:

```python
OLLAMA_CONFIG = {
    'base_url': 'http://127.0.0.1:11434',
    'default_model': 'gemma3n:latest',  # Change this if needed
    'generation_params': {
        'temperature': 0.7,
        'num_predict': 1000,
        'top_p': 0.9
    }
}
```

### Alternative Models

If you prefer to use a different model, you can pull and configure alternatives:

```powershell
# Alternative models (smaller, faster)
ollama pull llama3.2:3b
ollama pull gemma3:latest

# Larger models (more capable, slower)
ollama pull llama3:8b
ollama pull gemma3:7b
```

## 🛠️ Troubleshooting

### Common Issues

**1. "Model not found" error**
```bash
# Solution: Pull the model
ollama pull gemma3n:latest
```

**2. "Ollama not available" error**
```bash
# Solution: Start Ollama service
ollama serve
```

**3. Slow model responses**
- The first request may be slow as the model loads into memory
- Subsequent requests should be faster
- Consider using a smaller model for faster responses

**4. Port already in use**
- Change the port in `app.py` if 7860 is occupied
- Or stop other services using that port

### Performance Tips

- **First Run**: The first AI recommendation may take 30-60 seconds as the model loads
- **Memory**: Ensure you have at least 8GB RAM for optimal performance
- **Storage**: Keep at least 10GB free space for model storage

## 📞 Support

If you encounter issues:

1. **Check Ollama Status**: `ollama list` and `ollama serve`
2. **Verify Model**: Ensure `gemma3n:latest` is installed
3. **Check Logs**: Look at the application console output for error messages
4. **Restart Services**: Try restarting both Ollama and the application

## 🌟 Contributing

Contributions are welcome! Please feel free to submit issues and enhancement requests.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Happy Organic Farming! 🌱🚜**