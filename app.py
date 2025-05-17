from flask import Flask, request, jsonify
from transformers import RobertaTokenizer, RobertaModel
import torch
import torch.nn as nn
import re
import urllib.parse
import numpy as np
import joblib
import pandas as pd
from collections import OrderedDict

app = Flask(__name__)

# Define the model architecture directly in the Flask app
class RobertaURLClassifier(nn.Module):
    def __init__(self, n_classes=2, n_features=75):  # Update n_features based on your actual feature count
        super(RobertaURLClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.drop = nn.Dropout(p=0.3)
        
        # Attention layer for text
        self.text_attention = nn.Sequential(
            nn.Linear(768, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )
        
        # Feature processing
        self.feature_net = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Combined classifier
        self.fc = nn.Linear(768 + 128, n_classes)
    
    def forward(self, input_ids, attention_mask, features):
        # Text processing
        text_outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        last_hidden_state = text_outputs.last_hidden_state
        attention_weights = self.text_attention(last_hidden_state)
        text_vector = torch.sum(attention_weights * last_hidden_state, dim=1)
        text_vector = self.drop(text_vector)
        
        # Feature processing
        feature_vector = self.feature_net(features)
        
        # Combine
        combined = torch.cat([text_vector, feature_vector], dim=1)
        return self.fc(combined)

# Load all necessary artifacts
print("Loading tokenizer...")
tokenizer = RobertaTokenizer.from_pretrained('roberta_tokenizer')

print("Loading scaler...")
scaler = joblib.load('url_feature_scaler.joblib')

print("Loading class weights...")
class_weights = joblib.load('class_weights.joblib')

print("Initializing model...")
model = RobertaURLClassifier()
print("Loading model weights...")
model.load_state_dict(torch.load('phishing_url_roberta_final.pth', map_location=torch.device('cpu')))
model.eval()
print("Model loaded successfully!")

# Define suspicious words and TLDs (must match training)
suspicious_words = ['login', 'verify', 'secure', 'account', 'update', 'bank', 'signin',
                   'confirm', 'password', 'webscr', 'ebayisapi', 'paypal', 'billing',
                   'submit', 'security', 'validate', 'authentication', 'support',
                   'alert', 'unlock', 'reset', 'identity', 'recovery', 'limited',
                   'service', 'access', 'authorize', 'credentials', 'payment', 'urgent',
                   'message', 'warning', 'win', 'free', 'bonus', 'click', 'verifyemail',
                   'suspend', 'locked', 'danger', 'checkout', 'invoice', 'order']

common_tlds = ['com', 'org', 'net', 'edu', 'gov', 'co', 'io', 'info']

def extract_url_features(url):
    """Extract multiple features from URLs for enhanced detection."""
    features = OrderedDict()  # Using OrderedDict to maintain feature order
    url = url.lower()
    
    # Basic features
    features['url_length'] = len(url)
    parsed = urllib.parse.urlparse(url)
    domain = parsed.netloc
    features['domain_length'] = len(domain)

    # Character counts
    chars_to_count = ['.', '-', '_', '/', '?', '=', '@', '&', '!', ' ', '%', 
                     '~', ',', '+', '*', '#', '$']
    for char in chars_to_count:
        key = f'num_{char.replace(".", "dot").replace("-", "hyphen").replace("_", "underscore")}'
        features[key] = url.count(char)

    # Protocol features
    features['has_https'] = int(url.startswith('https'))
    features['has_http'] = int(url.startswith('http'))

    # Suspicious words
    for word in suspicious_words:
        features[f'contains_{word}'] = int(word in url)

    # TLD features
    for tld in common_tlds:
        features[f'has_{tld}'] = int(domain.endswith(f'.{tld}'))

    # Digit counts
    features['num_digits_in_url'] = sum(c.isdigit() for c in url)
    features['num_digits_in_domain'] = sum(c.isdigit() for c in domain)

    # Entropy calculation
    if domain:
        char_counts = {}
        for char in domain:
            char_counts[char] = char_counts.get(char, 0) + 1

        entropy = 0
        for count in char_counts.values():
            prob = count / len(domain)
            entropy -= prob * np.log2(prob)
        features['domain_entropy'] = entropy
    else:
        features['domain_entropy'] = 0

    return features

def preprocess_url(url):
    url = re.sub(r'^https?:\/\/', '', url)
    url = re.sub(r'^www\.', '', url)
    url = re.sub(r'\?.*$', '', url)
    url = re.sub(r'\/$', '', url)
    return url

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({'error': 'No URL provided'}), 400
    
    url = data['url']
    
    try:
        # 1. Preprocess URL
        processed_url = preprocess_url(url)
        
        # 2. Extract features
        features = extract_url_features(url)
        
        # 3. Prepare features DataFrame
        features_df = pd.DataFrame([features])
        
        # 4. Scale numerical features
        numeric_cols = [col for col in features_df.columns if col not in 
                       [f'contains_{w}' for w in suspicious_words] + 
                       [f'has_{tld}' for tld in common_tlds] + 
                       ['has_https', 'has_http']]
        features_df[numeric_cols] = scaler.transform(features_df[numeric_cols])
        
        # 5. Convert to tensor
        features_tensor = torch.tensor(features_df.values, dtype=torch.float)
        
        # 6. Tokenize text
        encoding = tokenizer.encode_plus(
            processed_url,
            add_special_tokens=True,
            max_length=128,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        # 7. Make prediction
        with torch.no_grad():
            outputs = model(
                input_ids=encoding['input_ids'],
                attention_mask=encoding['attention_mask'],
                features=features_tensor
            )
            probs = torch.softmax(outputs, dim=1)
            _, prediction = torch.max(outputs, dim=1)
        
        is_phishing = bool(prediction.item() == 1)
        confidence = probs[0][prediction.item()].item()
        
        return jsonify({
            'url': url,
            'isPhishing': is_phishing,
            'confidence': round(confidence, 4),
            'features': features_df.iloc[0].to_dict()  # Return processed features
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'Error processing URL',
            'url': url
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'components': {
            'tokenizer': True,
            'scaler': True,
            'model_weights': True
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)