from flask import Flask, request, render_template, redirect, url_for
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained Random Forest model
model = joblib.load("random_forest_model.pkl")

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/form')
def home():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data
        data = request.form
        
        # Process inputs with better error handling
        def safe_int(value, default=0):
            try:
                return int(value) if value != '' else default
            except (ValueError, TypeError):
                return default
                
        def safe_float(value, default=0.0):
            try:
                return float(value) if value != '' else default
            except (ValueError, TypeError):
                return default
        
        # Process inputs including new fields with safe conversion
        lead_time = safe_int(data.get('lead_time'))
        special_requests = safe_int(data.get('special_request'))
        reservation_month = safe_int(data.get('reservation_month', 1))
        booking_id = safe_int(data.get('Booking_ID'))
        num_adults = safe_int(data.get('no_of_adult'))
        num_children = safe_int(data.get('No_of_children'))
        avg_price = safe_float(data.get('average_price'))
        
        # Process room type - map from string to int
        room_type_mapping = {
            'Room_Type 1': 1, 'Room_Type 2': 2, 'Room_Type 3': 3, 
            'Room_Type 4': 4, 'Room_Type 5': 5, 'Room_Type 6': 6, 'Room_Type 7': 7
        }
        room_type = room_type_mapping.get(data.get('room_type', 'Room_Type 1'), 1)
        
        # Process boolean inputs
        regular_customer = 1 if data.get('regular_customer') == 'on' else 0
        car_parking_space = 1 if data.get('car_parking') == '1' else 0
        is_weekend = 1 if data.get('is_weekend') == 'on' else 0
        weekend_days = safe_int(data.get('weekend_day_input')) if is_weekend else 0
        
        # Calculate percent_can from cancellation history with safe conversion
        prev_canceled = safe_int(data.get('prev_canceled'))
        prev_not_canceled = safe_int(data.get('prev_not_canceled'))
        # Avoid division by zero
        total_reservations = prev_canceled + prev_not_canceled
        percent_can = prev_canceled / (total_reservations + 1e-10) if total_reservations > 0 else 0
        
        # Process market segment
        market_segment = data.get('market_segment', 'Online')
        market_features = {
            'Market_Complementary': 1 if market_segment == 'Complementary' else 0,
            'Market_Corporate': 1 if market_segment == 'Corporate' else 0,
            'Market_Offline': 1 if market_segment == 'Offline' else 0,
            'Market_Online': 1 if market_segment == 'Online' else 0
        }
        
        # Process meal type
        meal_type = data.get('meal_type', 'Meal Plan 1')
        meal_features = {
            'MealType_Meal Plan 1': 1 if meal_type == 'Meal Plan 1' else 0,
            'MealType_Meal Plan 2': 1 if meal_type == 'Meal Plan 2' else 0
        }
        
        # Create feature array with original features
        feature_names = ['car parking space', 'room type', 'lead time', 'regular customer', 
                        'special requests', 'reservation_month', 'Market_Complementary', 
                        'Market_Corporate', 'Market_Offline', 'Market_Online', 
                        'MealType_Meal Plan 1', 'MealType_Meal Plan 2', 
                        'percent_can', 'is_weekend']
        
        features_dict = {
            'car parking space': car_parking_space,
            'room type': room_type,
            'lead time': lead_time,
            'regular customer': regular_customer,
            'special requests': special_requests,
            'reservation_month': reservation_month,
            'Market_Complementary': market_features['Market_Complementary'],
            'Market_Corporate': market_features['Market_Corporate'],
            'Market_Offline': market_features['Market_Offline'],
            'Market_Online': market_features['Market_Online'],
            'MealType_Meal Plan 1': meal_features['MealType_Meal Plan 1'],
            'MealType_Meal Plan 2': meal_features['MealType_Meal Plan 2'],
            'percent_can': percent_can,
            'is_weekend': is_weekend
        }
        
        # Create feature array in the correct order
        features = [features_dict[name] for name in feature_names]
        
        # Reshape for prediction
        features_array = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features_array)[0]
        
        # Convert prediction to text
        result = "Not Cancelled" if prediction else "Cancelled"
        
        return render_template("index.html", prediction=result)
    
    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
