from TextClassifier.views import Predict
from django.conf.urls import url

app_name = 'TextClassifier'
urlpatterns = [
    url(r'^getCategory/$', Predict.as_view(), name="predict"),
    ]