
from django.shortcuts import render
from django.http import JsonResponse
from .ml_tools import model_Load as ml
from .ml_tools import feature_extraction as fe
import requests as re
from bs4 import BeautifulSoup

def index(request):
    context = {}
    if request.method == 'POST':
        url = request.POST.get('url')
        model_name = request.POST.get('model')
        context['url'] = url
        context['model_name'] = model_name
        try:
            response = re.get(url, verify=False, timeout=10)
            if response.status_code == 404:
                context['status'] = 'success'
                context['prediction'] = " 404!! This Website doesn't exist anymore"
                context['color'] = 'danger'
            elif response.status_code == 200:
                soup = BeautifulSoup(response.content, "html.parser")
                vector = [fe.create_vector(soup)] 
                models = ml.load_models()
                selected_model = models.get(model_name, models['Random Forest'])  
                result = ml.predict(selected_model, vector)
                if result[0] == 0:
                    prediction = "This web page seems legitimate!"
                    color = 'success'
                else:
                    prediction = "Attention! This web page is a potential PHISHING!"
                    color = 'danger'
                context['status'] = 'success'
                context['prediction'] = prediction
                context['color'] = color
            else:
                context['status'] = 'error'
                context['message'] = 'HTTP request failed.'
        except re.exceptions.RequestException as e:
            context['status'] = 'error'
            context['message'] = str(e)
            context['color'] = 'danger'
    return render(request, 'index.html', context)

def more_info(request):
    return render(request, 'more_info.html')

from django.shortcuts import render, redirect
from django.core.mail import send_mail
from django.conf import settings
from .forms import FeedbackForm

def feedback_view(request):
    if request.method == 'POST':
        form = FeedbackForm(request.POST)
        if form.is_valid():
            feedback_type = form.cleaned_data['feedback_type']
            name = form.cleaned_data['name']
            email = form.cleaned_data['email']
            feedback = form.cleaned_data['feedback']
            
            # Compose the email content
            subject = f'New Feedback: {feedback_type}'
            message = f'Name: {name}\nEmail: {email}\n\nFeedback:\n{feedback}'
            from_email = settings.EMAIL_HOST_USER
            recipient_list = [settings.EMAIL_HOST_USER]
            
            # Send the email
            send_mail(subject, message, from_email, recipient_list)
            
            return redirect('thank_you')
    else:
        form = FeedbackForm()
    return render(request, 'form.html', {'form': form})


