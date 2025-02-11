from django.shortcuts import render, redirect
from django.core.files.base import ContentFile
from.models import Book, CVmodel
from .forms import ImageForm, WeightsForm
from .cam import Grad_CAM
from .cnn import Net
import numpy as np
import os
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from io import BytesIO

def caluculate_cam(original_image, weight):
    net = Net("tf_efficientnet_b0")
    cam = Grad_CAM(net, weight, net.model.conv_head)
    cam_img = cam(original_image)

    return cam_img

def numpy2image(cam_image, original_file_name):
    name, ext = os.path.splitext(original_file_name)
    new_filename = f"{name}_cam{ext}"

    img = Image.fromarray((cam_image * 255).astype(np.uint8))
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='PNG') 
    img_byte_arr.seek(0) 
    
    return ContentFile(img_byte_arr.read(), name=new_filename)

def upload(request):
    if request.method == "POST":
        image_form = ImageForm(request.POST)
        weight_form = WeightsForm(request.POST)
        if image_form.is_valid():
            cv_model = CVmodel()
            # cv_model.title = request.POST['title']
            cv_model.weight_file = request.FILES['weight_file']

            book = Book()
            book.image = request.FILES['image']
            model_input = np.array(Image.open(request.FILES['image']), dtype = np.float32)
            model_input = torch.from_numpy(model_input).unsqueeze(0).unsqueeze(1)
            model_input /= 255

            cam_img = caluculate_cam(model_input, request.FILES['weight_file'])

            book.cam_image = numpy2image(cam_img, book.image.name)

            cv_model.save()
            book.save()

            return redirect("image_detail", id = book.id)
    else:
        image_form = ImageForm()
        weight_form = WeightsForm()
        all_objects = Book.objects.all()
        ids = all_objects.values_list('id', flat=True)
    context = {"form1": image_form, "form2": weight_form, "ids": ids}
    return render(request, "blog/post_list.html", context)

def image_detail(request, id = 50):
    book = Book.objects.get(id=id)  
    context = {"original_image": book.image, "cam_image": book.cam_image}
    return render(request, 'blog/image_detail.html', context)