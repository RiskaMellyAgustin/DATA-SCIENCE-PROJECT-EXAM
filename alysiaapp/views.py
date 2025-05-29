from django.shortcuts import render
from .forms import UserInputForm
from .ml_scripts.predict_model import predict_waktu_belajar, predict_proba_waktu_belajar
from .ml_scripts.predict_cluster import predict_best_study_time

def input_form_view(request):
    recommendation = None
    proba_data = None
    show_chart = False
    cluster_id = None

    if request.method == 'POST':
        form = UserInputForm(request.POST)
        if form.is_valid():
            data = form.cleaned_data
            algorithm = data['algorithm']

            input_data = {
                'gender': data['gender'],
                'age': data['age'],
                'grade': data['grade'],
                'activity_type': data['activity_type'],
                'duration_minutes': data['duration_minutes'],
                'day_of_week': int(data['day_of_week'])
            }

            if algorithm == 'classification':
                recommendation = predict_waktu_belajar(input_data)
                proba_data = predict_proba_waktu_belajar(input_data)
                show_chart = True
                cluster_id = None

            elif algorithm == 'clustering':
                cluster_id, recommendation = predict_best_study_time(
                    input_data['gender'],
                    input_data['age'],
                    input_data['grade'],
                    input_data['activity_type'],
                    input_data['duration_minutes'],
                    input_data['day_of_week'],
                    None  # model akan diload otomatis di dalam fungsi
                )
                proba_data = None
                show_chart = False

    else:
        form = UserInputForm()

    context = {
        'form': form,
        'recommendation': recommendation,
        'proba_data': proba_data,
        'show_chart': show_chart,
        'cluster_id': cluster_id
    }
    return render(request, 'input_form.html', context)