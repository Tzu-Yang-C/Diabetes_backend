from django.shortcuts import render
import joblib
import os
from rest_framework.decorators import api_view,permission_classes
from rest_framework.response import Response
from django.views.decorators.csrf import csrf_exempt
from rest_framework.permissions import AllowAny

model_path = os.path.join(os.path.dirname(__file__), 'model', 'diabetes_dataset_model4.joblib')
model = joblib.load(model_path)
@csrf_exempt
@api_view(['POST'])
@permission_classes([AllowAny])
def predict_view(request):
    try:
        data = request.data  
        features = [[
            int(data['gender']),
            float(data['age']),
            int(data['hypertension']),
            int(data['heart_disease']),
            int(data['smoking_history']),
            float(data['bmi']),
            float(data['blood_glucose_level']),
        ]]
        prediction = model.predict_proba(features)[:, 1]
        prediction_list = prediction.tolist()
        return Response({'probabilities': prediction_list[0]})
    except Exception as e:
        return Response({'error': str(e)}, status=400)