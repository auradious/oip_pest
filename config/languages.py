"""
Multilingual configuration for Organic Farm Pest Management AI System
"""

# Language configurations
LANGUAGES = {
    'en': {
        'name': 'English',
        'flag': '🇺🇸',
        'ui': {
            'title': '🌱 Organic Farm Pest Management AI',
            'description': 'Upload an image of a pest to get instant identification and organic treatment recommendations',
            'upload_section': '📸 Upload Pest Image',
            'upload_label': 'Drag and drop or click to upload',
            'identify_button': '🔍 Identify Pest',
            'results_section': '🎯 Identification Results',
            'identification_label': 'Pest Identification',
            'treatment_label': '🌿 Treatment Recommendations',
            'tips_title': '📝 Photography Tips',
            'tips_content': '''**For best identification results:**
- Take clear, close-up photos
- Ensure good lighting
- Focus on the pest, minimize background
- Multiple angles can help accuracy
- Avoid blurry or dark images''',
            'footer_title': '🌱 Organic Farm Pest Management AI',
            'footer_subtitle': 'Helping farmers protect crops while preserving beneficial insects',
            'footer_disclaimer': 'Always consult local agricultural experts for comprehensive pest management',
            'language_selector': 'Language'
        },
        'predictions': {
            'no_image': '📸 **Please upload an image**',
            'no_image_desc': 'Upload a clear photo of the pest for identification.',
            'model_not_loaded': '❌ **Model not loaded**',
            'model_not_loaded_desc': '''**Demo Mode**

The AI model needs to be trained before making predictions. Once trained, you'll get real pest identification and treatment recommendations.''',
            'uncertain': '🤔 **Uncertain Identification**',
            'uncertain_desc': '⚠️ Low confidence - consider taking a clearer photo',
            'identified': '🔍 **Pest Identified**',
            'species': '**Species:**',
            'confidence': '**Confidence:**',
            'most_likely': 'Most likely:',
            'error': '❌ **Error during prediction:**',
            'error_desc': 'Please try again with a different image.',
            'recommendation_unclear': '**Recommendation:** Take a clearer, closer photo in good lighting for better identification.',
            'no_treatment': '✅ **No treatment needed** - This appears to be a beneficial insect!',
            'urgency_level': 'Urgency Level:',
            'economic_impact': 'Economic Impact:',
            'action_priority': '🚨 Action Priority:',
            'immediate_action': '🔥 **IMMEDIATE ACTION REQUIRED**',
            'action_recommended': '⚡ **ACTION RECOMMENDED**',
            'monitor_situation': '📝 **MONITOR SITUATION**'
        },
        'treatments': {
            'organic_treatments': '🌿 **Organic Treatments:**',
            'beetle': '• Neem oil spray\n• Beneficial nematodes\n• Row covers\n• Hand picking\n• Diatomaceous earth',
            'catterpillar': '• Bacillus thuringiensis (Bt)\n• Row covers\n• Hand picking\n• Beneficial wasps\n• Companion planting with herbs',
            'earwig': '• Beer traps\n• Diatomaceous earth\n• Remove garden debris\n• Beneficial predators\n• Copper strips',
            'grasshopper': '• Row covers\n• Beneficial birds habitat\n• Neem oil\n• Kaolin clay spray\n• Timing of plantings',
            'moth': '• Pheromone traps\n• Bacillus thuringiensis (Bt)\n• Row covers during flight season\n• Beneficial parasitic wasps\n• Light traps',
            'slug': '• Beer traps\n• Copper barriers\n• Diatomaceous earth\n• Iron phosphate baits\n• Remove hiding places',
            'snail': '• Beer traps\n• Copper barriers\n• Diatomaceous earth\n• Hand picking\n• Crushed eggshells',
            'wasp': '• Usually beneficial! Only treat if problematic\n• Remove food sources\n• Seal nest entrances\n• Professional removal if needed',
            'weevil': '• Beneficial nematodes\n• Diatomaceous earth\n• Remove infected plants\n• Crop rotation\n• Sticky traps'
        }
    },
    'vi': {
        'name': 'Tiếng Việt',
        'flag': '🇻🇳',
        'ui': {
            'title': '🌱 AI Quản Lý Sâu Bệnh Nông Nghiệp Hữu Cơ',
            'description': 'Tải lên hình ảnh sâu bệnh để nhận diện ngay lập tức và nhận khuyến nghị điều trị hữu cơ',
            'upload_section': '📸 Tải Lên Hình Ảnh Sâu Bệnh',
            'upload_label': 'Kéo thả hoặc nhấp để tải lên',
            'identify_button': '🔍 Nhận Diện Sâu Bệnh',
            'results_section': '🎯 Kết Quả Nhận Diện',
            'identification_label': 'Nhận Diện Sâu Bệnh',
            'treatment_label': '🌿 Khuyến Nghị Điều Trị',
            'tips_title': '📝 Mẹo Chụp Ảnh',
            'tips_content': '''**Để có kết quả nhận diện tốt nhất:**
- Chụp ảnh rõ nét, cận cảnh
- Đảm bảo ánh sáng tốt
- Tập trung vào sâu bệnh, giảm thiểu nền
- Nhiều góc độ có thể giúp tăng độ chính xác
- Tránh ảnh mờ hoặc tối''',
            'footer_title': '🌱 AI Quản Lý Sâu Bệnh Nông Nghiệp Hữu Cơ',
            'footer_subtitle': 'Giúp nông dân bảo vệ cây trồng đồng thời bảo tồn côn trùng có ích',
            'footer_disclaimer': 'Luôn tham khảo ý kiến chuyên gia nông nghiệp địa phương để quản lý sâu bệnh toàn diện',
            'language_selector': 'Ngôn ngữ'
        },
        'predictions': {
            'no_image': '📸 **Vui lòng tải lên hình ảnh**',
            'no_image_desc': 'Tải lên ảnh rõ nét của sâu bệnh để nhận diện.',
            'model_not_loaded': '❌ **Mô hình chưa được tải**',
            'model_not_loaded_desc': '''**Chế Độ Demo**

Mô hình AI cần được huấn luyện trước khi đưa ra dự đoán. Sau khi huấn luyện, bạn sẽ nhận được nhận diện sâu bệnh thực tế và khuyến nghị điều trị.''',
            'uncertain': '🤔 **Nhận Diện Không Chắc Chắn**',
            'uncertain_desc': '⚠️ Độ tin cậy thấp - hãy xem xét chụp ảnh rõ hơn',
            'identified': '🔍 **Đã Nhận Diện Sâu Bệnh**',
            'species': '**Loài:**',
            'confidence': '**Độ Tin Cậy:**',
            'most_likely': 'Có thể là:',
            'error': '❌ **Lỗi trong quá trình dự đoán:**',
            'error_desc': 'Vui lòng thử lại với hình ảnh khác.',
            'recommendation_unclear': '**Khuyến nghị:** Chụp ảnh rõ hơn, cận cảnh hơn trong ánh sáng tốt để nhận diện tốt hơn.',
            'no_treatment': '✅ **Không cần điều trị** - Đây có vẻ là côn trùng có ích!',
            'urgency_level': 'Mức Độ Khẩn Cấp:',
            'economic_impact': 'Tác Động Kinh Tế:',
            'action_priority': '🚨 Ưu Tiên Hành Động:',
            'immediate_action': '🔥 **CẦN HÀNH ĐỘNG NGAY LẬP TỨC**',
            'action_recommended': '⚡ **KHUYẾN NGHỊ HÀNH ĐỘNG**',
            'monitor_situation': '📝 **THEO DÕI TÌNH HÌNH**'
        },
        'treatments': {
            'organic_treatments': '🌿 **Phương Pháp Điều Trị Hữu Cơ:**',
            'beetle': '• Xịt dầu neem\n• Tuyến trùng có ích\n• Lưới che phủ\n• Bắt tay\n• Đất diatomaceous',
            'catterpillar': '• Bacillus thuringiensis (Bt)\n• Lưới che phủ\n• Bắt tay\n• Ong ký sinh có ích\n• Trồng xen với thảo mộc',
            'earwig': '• Bẫy bia\n• Đất diatomaceous\n• Dọn dẹp mảnh vụn vườn\n• Động vật săn mồi có ích\n• Dải đồng',
            'grasshopper': '• Lưới che phủ\n• Môi trường sống cho chim có ích\n• Dầu neem\n• Xịt đất sét kaolin\n• Thời gian trồng trọt',
            'moth': '• Bẫy pheromone\n• Bacillus thuringiensis (Bt)\n• Lưới che trong mùa bay\n• Ong ký sinh có ích\n• Bẫy đèn',
            'slug': '• Bẫy bia\n• Rào cản đồng\n• Đất diatomaceous\n• Mồi iron phosphate\n• Loại bỏ nơi ẩn náu',
            'snail': '• Bẫy bia\n• Rào cản đồng\n• Đất diatomaceous\n• Bắt tay\n• Vỏ trứng nghiền',
            'wasp': '• Thường có ích! Chỉ điều trị nếu có vấn đề\n• Loại bỏ nguồn thức ăn\n• Bịt kín lối vào tổ\n• Loại bỏ chuyên nghiệp nếu cần',
            'weevil': '• Tuyến trùng có ích\n• Đất diatomaceous\n• Loại bỏ cây bị nhiễm\n• Luân canh cây trồng\n• Bẫy dính'
        }
    }
}

# Default language
DEFAULT_LANGUAGE = 'en'

# Available languages for dropdown
AVAILABLE_LANGUAGES = [(lang_data['flag'] + ' ' + lang_data['name'], lang_code) 
                       for lang_code, lang_data in LANGUAGES.items()]