from django.shortcuts import render

# Create your views here.

def home_screen_view(request):
    
    context = {}
    context['some_string'] = "this is a string i'm passing to the view"

    list_of_values = []
    list_of_values.append("1st entry")
    list_of_values.append("2nd entry")
    list_of_values.append("3rd entry")
    list_of_values.append("4th entry")
    context['list_of_values'] = list_of_values

    return render(request, "personal/home.html", context)

def upload_page_view(request):
    context = {}
    return render(request, 'personal/upload_page.html', context)

def about(request):
    return render(request, 'personal/about.html')