from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.shortcuts import redirect
from .form import ImageForm
from .models import Image
from subprocess import run, PIPE
import sys
import os
from django.conf import settings
import time
# Create your views here.
def index(request):
	return render(request,'first.html',{
        'namaste':"Namaste!",
        'next':"Welcome To Change Detection",
        'inst':"http://127.0.0.1:8000/upload",
        'sentence':"Please click ",
        'final': " to upload videos"
    })



def video_list(request):
    images = Image.objects.all()
    if request.method == "POST":
        return render(request, 'img_list.html',{
            'images':images,
            'inst':"http://127.0.0.1:8000/upload",
        })
    return render(request, 'img_list.html',{
            'images':images,
            'inst':"http://127.0.0.1:8000/upload",
        })

def image_upload(request):
    images = Image.objects.all()

    if request.method == "POST":
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('img_upload')
    else:
        form = ImageForm()
    return render(request, 'img_upload.html',{
        'form': form,
        'images':images,
        'home':"http://127.0.0.1:8000/",
    })

def delete_image(request, pk):
    if request.method=="POST":
        image = Image.objects.get(pk=pk)
        image.delete()
        return redirect('img_list')
    return render(request,'img_list.html')

def detect(request):
    start_time = time.time()
    #inp = request.POST.get('param')
    out = run([sys.executable,"./changedetection/code.py"], shell = False, stdout= PIPE)
    end_time = time.time()
    time_taken = end_time-start_time
    temp = str(out.stdout)[2:]
    temp = temp.replace('\\r','')
    temp = temp.replace('\\n','')
    temp = temp.replace('\'','')
    #print("Time taken",end_time-start_time)
    #print(out)

    return render(request, 'date.html',{
        'data1':temp,
        'data2':time_taken,
        'upload':'http://127.0.0.1:8000/upload'

    })

def delete_all(request):
    frames_video1 = './media/media/frames/video1/'
    frames_video2 = './media/media/frames/video2/'
    key_frames_video1 = './media/media/key-frames/video1/'
    key_frames_video2 = './media/media/key-frames/video2/'
    panoramas = './media/media/result/'
    output = './static/images/'
    folder = [frames_video1, frames_video2, key_frames_video1, key_frames_video2, panoramas,output]
    for folder_name in folder:
        filelist = [ f for f in os.listdir(folder_name) if f.endswith(".jpg") ]
        for f in filelist:
            os.remove(os.path.join(folder_name, f))
    return redirect('img_upload')
