"""
Pest Prediction Logic
Handles AI model loading and prediction processing
"""

import numpy as np
import json
import sys
from pathlib import Path
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.config import *
from config.languages import LANGUAGES, DEFAULT_LANGUAGE
from presentation.ollama_service import OllamaService

class PestPredictor:
    """
    Handles pest identification and treatment recommendations
    """
    
    def __init__(self):
        self.model = None
        self.class_names = {}
        self.ollama_service = OllamaService()
        self.load_model_and_classes()
        
    def load_model_and_classes(self):
        """
        Load the trained model and class mappings
        """
        try:
            # Try to load TensorFlow and model
            import tensorflow as tf
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
            
            # Load trained model with custom objects
            model_path = MODEL_PATHS['best_model']
            print(f"🔍 Looking for model at: {model_path}")
            
            if model_path.exists():
                try:
                    # Define custom objects for model loading
                    custom_objects = {
                        'mobilenet_preprocess': mobilenet_preprocess,
                        'preprocess_input': mobilenet_preprocess,
                    }
                    
                    # Load model with custom objects scope
                    print(f"📥 Loading model with custom objects...")
                    with tf.keras.utils.custom_object_scope(custom_objects):
                        self.model = tf.keras.models.load_model(model_path)
                    print(f"✅ Model loaded successfully from {model_path}")
                except Exception as e:
                    print(f"❌ Error loading model: {e}")
                    print(f"💡 This might be due to missing custom objects or incompatible model format.")
                    print(f"🔧 Try retraining the model or check the model file integrity.")
                    self.model = None
            else:
                print(f"❌ Model file not found at: {model_path}")
                print(f"📁 Available files in models directory:")
                models_dir = model_path.parent
                if models_dir.exists():
                    for file in models_dir.iterdir():
                        print(f"   - {file.name}")
                else:
                    print(f"   - Models directory doesn't exist")
                print(f"🔧 Please ensure you have trained the model using:")
                print(f"   1. python src/data_preprocessing.py")
                print(f"   2. python src/mnhybrid_training.py")
                self.model = None
                
            # Load class mappings
            class_names_path = MODEL_PATHS['class_names']
            if class_names_path.exists():
                with open(class_names_path, 'r') as f:
                    self.class_names = json.load(f)
                print(f"✅ Class names loaded from {class_names_path}")
            else:
                print(f"❌ Class names not found at {class_names_path}")
                # Fallback to config classes
                self.class_names = {
                    'idx_to_class': {str(i): cls for i, cls in enumerate(HARMFUL_PEST_CLASSES)}
                }
                
        except ImportError:
            print("❌ TensorFlow not available - running in demo mode")
            self.model = None
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None
    
    def preprocess_image(self, image):
        """
        Preprocess uploaded image for model prediction
        Must match the preprocessing used during training!
        """
        if image is None:
            return None
            
        try:
            # Convert to PIL Image if needed
            if not isinstance(image, Image.Image):
                try:
                    image = Image.fromarray(image)
                except Exception as e:
                    raise ValueError(f"Invalid image format: {str(e)}")
            
            # Resize to model input size (224, 224 for MobileNet)
            target_size = (224, 224)  # MobileNet standard size
            image = image.resize(target_size)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to numpy array - IMPORTANT: Keep in [0,255] range for MobileNet!
            img_array = np.array(image, dtype=np.float32)
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            
            # MobileNet preprocessing will be handled by the model internally
            # DO NOT normalize to [0,1] here - that's done by mobilenet_preprocess in the model
            
            print(f"🔍 Preprocessed image shape: {img_array.shape}")
            print(f"🔍 Image value range: [{img_array.min():.1f}, {img_array.max():.1f}]")
            
            return img_array
        except Exception as e:
            raise ValueError(f"Error preprocessing image: {str(e)}")
    
    def get_treatment_recommendation(self, pest_class, confidence=0.0, language='en'):
        """
        Get organic treatment recommendations for identified insect
        Uses Ollama AI as primary source, falls back to static recommendations
        """
        # Check if insect is beneficial (protection recommendations instead of treatment)
        if pest_class in BENEFICIAL_CLASSES:
            lang_data = LANGUAGES.get(language, LANGUAGES[DEFAULT_LANGUAGE])
            return self._get_beneficial_insect_recommendation(pest_class, confidence, language)
        
        # For harmful pests, try Ollama first for AI-powered recommendations
        try:
            if self.ollama_service.is_available:
                print("🤖 Generating AI-powered treatment recommendation...")
                ai_recommendation = self.ollama_service.generate_treatment_recommendation(
                    pest_class, confidence
                )
                return ai_recommendation
            else:
                print("⚠️ Ollama unavailable, using static recommendations...")
                return self._get_static_recommendation(pest_class, language)
        except Exception as e:
            print(f"❌ Error with Ollama service: {e}")
            print("🔄 Falling back to static recommendations...")
            return self._get_static_recommendation(pest_class, language)
    
    def _get_beneficial_insect_recommendation(self, insect_class, confidence, language='en'):
        """
        Get protection recommendations for beneficial insects
        """
        lang_data = LANGUAGES.get(language, LANGUAGES[DEFAULT_LANGUAGE])
        
        # Get insect characteristics
        economic_benefit = abs(ECONOMIC_IMPACT.get(insect_class, 3))  # Convert negative to positive for display
        protection_action = TREATMENT_URGENCY.get(insect_class, 'Protect')
        
        # Beneficial insect specific recommendations
        beneficial_recommendations = {
            'ants': """🌟 **Beneficial Pest Controller Detected**
**Ants** are natural pest controllers that can help manage harmful insects in your garden.

**Protection Recommendations:**
• 🌱 **Preserve their habitat** - avoid disturbing ant colonies near garden areas
• 🚫 **Avoid broad-spectrum pesticides** that could harm beneficial ants
• 🍯 **Provide food sources** - plant flowers that produce nectar
• 🏠 **Create shelter** - leave some undisturbed soil areas for nesting
• 💧 **Maintain moisture** - ensure adequate water sources nearby

**Benefits to Your Garden:**
• Control aphids, caterpillars, and other harmful pests
• Aerate soil through tunneling activities
• Distribute organic matter throughout the soil""",

            'bees': """🌟 **Critical Pollinator Detected**
**Bees** are essential pollinators crucial for crop production and ecosystem health.

**Protection Recommendations:**
• 🌸 **Plant bee-friendly flowers** throughout the growing season
• 🚫 **NEVER use pesticides** when bees are present
• 🏠 **Provide nesting sites** - leave bare soil patches and plant stems
• 💧 **Create water sources** - shallow dishes with landing spots
• 🌿 **Avoid lawn chemicals** that can harm bee colonies
• ⏰ **Time garden activities** to avoid peak bee activity hours

**Critical Importance:**
• Responsible for pollinating 1/3 of food crops
• Essential for fruit and vegetable production
• Support ecosystem biodiversity""",

            'earthworms': """🌟 **Soil Health Engineer Detected**
**Earthworms** are vital soil ecosystem engineers that improve growing conditions.

**Protection Recommendations:**
• 🌱 **Maintain organic matter** - add compost and leaf litter
• 💧 **Keep soil moist** but not waterlogged
• 🚫 **Avoid chemical fertilizers** that can harm worm populations
• 🌿 **Use organic mulch** to provide food and shelter
• 🔄 **Minimize soil disturbance** - reduce deep tilling
• 🍂 **Leave organic debris** for them to decompose

**Soil Benefits:**
• Improve soil structure and drainage
• Increase nutrient availability for plants
• Enhance water retention capacity
• Create natural fertilizer through castings"""
        }
        
        recommendation = beneficial_recommendations.get(insect_class, 
            f"""🌟 **Beneficial Insect Detected**
This {insect_class} provides positive benefits to your garden ecosystem.

**General Protection Guidelines:**
• 🌱 Maintain diverse plant species to support beneficial insects
• 🚫 Avoid broad-spectrum pesticides that harm beneficial species
• 🏠 Provide shelter through diverse garden structures
• 💧 Ensure adequate water sources
• 🌿 Use organic gardening practices""")
        
        # Add economic benefit information
        benefit_emojis = {1: "💚", 2: "💚💚", 3: "💚💚💚", 4: "💚💚💚💚", 5: "💚💚💚💚💚"}
        
        final_recommendation = f"""{recommendation}

{benefit_emojis.get(economic_benefit, "💚💚💚")} **Economic Benefit: {economic_benefit}/5**
🛡️ **Action Required: {protection_action}**

⚠️ **Important:** This is a beneficial species that should be protected and encouraged, not treated as a pest!"""
        
        return final_recommendation
    
    def _get_static_recommendation(self, pest_class, language='en'):
        """
        Get static treatment recommendations (fallback when Ollama is unavailable)
        """
        lang_data = LANGUAGES.get(language, LANGUAGES[DEFAULT_LANGUAGE])
        
        # Get pest characteristics
        economic_impact = ECONOMIC_IMPACT.get(pest_class, 3)
        urgency = TREATMENT_URGENCY.get(pest_class, 'Medium')
        
        # Get treatment text for the pest
        treatment_text = lang_data['treatments'].get(pest_class, '')
        organic_treatments = lang_data['treatments']['organic_treatments']
        
        treatment = f"{organic_treatments}\n{treatment_text}"
        
        # Add urgency and impact information
        impact_emojis = {1: "💚", 2: "💛", 3: "🧡", 4: "❤️", 5: "💔"}
        urgency_emojis = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
        
        # Get localized urgency text
        urgency_level = lang_data['predictions']['urgency_level']
        economic_impact_text = lang_data['predictions']['economic_impact']
        action_priority = lang_data['predictions']['action_priority']
        
        # Get action priority text based on urgency
        if urgency == 'High':
            priority_text = lang_data['predictions']['immediate_action']
        elif urgency == 'Medium':
            priority_text = lang_data['predictions']['action_recommended']
        else:
            priority_text = lang_data['predictions']['monitor_situation']
        
        recommendation = f"""**📋 Static Treatment Plan**
{urgency_emojis[urgency]} **{urgency_level} {urgency}**
{impact_emojis[economic_impact]} **{economic_impact_text} {economic_impact}/5**

{treatment}

**{action_priority}**
{priority_text}

⚠️ **Note:** For enhanced AI-powered recommendations, ensure Ollama is running with the configured model."""
        
        return recommendation
    
    def predict_pest(self, image, language='en'):
        """
        Main prediction function called by the interface
        """
        lang_data = LANGUAGES.get(language, LANGUAGES[DEFAULT_LANGUAGE])
        
        if image is None:
            return lang_data['predictions']['no_image'], lang_data['predictions']['no_image_desc']
        
        if self.model is None:
            return (f"{lang_data['predictions']['model_not_loaded']}\n\nPlease train the model first using:\n"
                   "1. `python src/data_preprocessing.py`\n"
                   "2. `python src/model_training.py`"), lang_data['predictions']['model_not_loaded_desc']
        
        try:
            # Preprocess image
            try:
                processed_image = self.preprocess_image(image)
                if processed_image is None:
                    return lang_data['predictions']['no_image'], lang_data['predictions']['no_image_desc']
            except ValueError as e:
                return f"{lang_data['predictions']['error']}\n\n{str(e)}", lang_data['predictions']['error_desc']
            try:
                processed_image = self.preprocess_image(image)
                if processed_image is None:
                    return lang_data['predictions']['no_image'], lang_data['predictions']['no_image_desc']
            except ValueError as e:
                return f"{lang_data['predictions']['error']}\n\n{str(e)}", lang_data['predictions']['error_desc']
            
            # Make prediction
            predictions = self.model.predict(processed_image, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            
            # Debug prediction values
            print(f"🔍 Raw predictions: {predictions[0]}")
            print(f"🔍 Predicted class index: {predicted_class_idx}")
            print(f"🔍 All class probabilities:")
            for i, prob in enumerate(predictions[0]):
                class_name = self.class_names['idx_to_class'].get(str(i), f'Class_{i}')
                print(f"   {class_name}: {prob:.4f} ({prob*100:.2f}%)")
            
            # Get class name
            predicted_class = self.class_names['idx_to_class'].get(str(predicted_class_idx), 'Unknown')
            print(f"🎯 Final prediction: {predicted_class} with {confidence*100:.2f}% confidence")
            
            # Format prediction result based on insect type
            confidence_percent = confidence * 100
            
            if predicted_class in BENEFICIAL_CLASSES:
                # Beneficial insect detected
                if confidence < 0.15:
                    result = f"{lang_data['predictions']['uncertain']}\n\n🌟 Most likely: **{predicted_class.title()}** (Beneficial)\n{lang_data['predictions']['confidence']} {confidence_percent:.1f}%\n\n{lang_data['predictions']['uncertain_desc']}"
                    recommendation = lang_data['predictions']['recommendation_unclear']
                else:
                    result = f"🌟 **Beneficial Insect Identified**\n\n**Species:** {predicted_class.title()}\n{lang_data['predictions']['confidence']} {confidence_percent:.1f}%\n\n✅ This is a helpful insect that benefits your garden!"
                    recommendation = self.get_treatment_recommendation(predicted_class, confidence, language)
            else:
                # Harmful pest detected
                if confidence < 0.15:
                    result = f"{lang_data['predictions']['uncertain']}\n\n🐛 Most likely: **{predicted_class.title()}** (Harmful)\n{lang_data['predictions']['confidence']} {confidence_percent:.1f}%\n\n{lang_data['predictions']['uncertain_desc']}"
                    recommendation = lang_data['predictions']['recommendation_unclear']
                else:
                    result = f"🐛 **Harmful Pest Identified**\n\n**Species:** {predicted_class.title()}\n{lang_data['predictions']['confidence']} {confidence_percent:.1f}%\n\n⚠️ Treatment may be needed to protect your crops."
                    recommendation = self.get_treatment_recommendation(predicted_class, confidence, language)
            
            return result, recommendation
            
        except Exception as e:
            return f"{lang_data['predictions']['error']} {str(e)}", lang_data['predictions']['error_desc']
