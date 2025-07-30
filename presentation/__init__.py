"""
Presentation package for Organic Farm Pest Management AI System
Contains all frontend interface components
"""

from .pest_interface import create_pest_management_interface
from .pest_predictor import PestPredictor

__all__ = ['create_pest_management_interface', 'PestPredictor']
