from django.shortcuts import render
import TextClassifier.mlmodelfile as ml
from rest_framework import views
from rest_framework import status
from rest_framework.response import Response

class Predict(views.APIView):
    def post(self, request):
        # To store the API response
        predictions = []

        # Extract the input mail body
        input = request.data.pop('mail_body')
        try:
            # Calling the model for prediction
            predictions = ml.machine_learning(input)

        # Error handling
        except Exception as err:
            return Response(str(err), status=status.HTTP_400_BAD_REQUEST)

        return Response(predictions, status=status.HTTP_200_OK)




