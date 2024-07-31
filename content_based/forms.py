from django import forms

class FeedbackForm(forms.Form):
    FEEDBACK_CHOICES = [
        ('bug', 'Report Bug'),
        ('suggestion', 'Suggestion'),
        ('experience', 'Share Experience')
    ]
    feedback_type = forms.ChoiceField(choices=FEEDBACK_CHOICES, widget=forms.RadioSelect)
    name = forms.CharField(max_length=100, required=False)
    email = forms.EmailField(required=False)
    feedback = forms.CharField(widget=forms.Textarea)

    def __init__(self, *args, **kwargs):
        super(FeedbackForm, self).__init__(*args, **kwargs)
        self.fields['name'].widget.attrs.update({'class': 'form-control', 'placeholder': 'Enter your name'})
        self.fields['email'].widget.attrs.update({'class': 'form-control', 'placeholder': 'Enter your email'})
        self.fields['feedback'].widget.attrs.update({'class': 'form-control', 'rows': '4', 'placeholder': 'Enter your feedback'})

