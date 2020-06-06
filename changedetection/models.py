from django.db import models
from imagekit.models import ImageSpecField
from imagekit.models import ProcessedImageField

from imagekit.processors import ResizeToFill

# Create your models here.
class Image(models.Model):
    title = models.CharField(max_length = 100,blank = True)
    pc = models.CharField(max_length=100,blank = True) 
    video1 = models.FileField(upload_to = "media/video/")
    video2 = models.FileField(upload_to = "media/video/" )
    #cover = models.ImageField(upload_to = "media/covers/", null= True, blank = True)
    # cover_thumbnail = ImageSpecField(source="cover",
    #                                         processors = [ResizeToFill(200,200)],
    #                                         format = "JPEG",
    #                                         options = {'quality':60})

    def __str__(self):
        return self.title

    def delete(self, *args, **kwargs):
        self.video1.delete()
        self.video2.delete()

        #self.cover.delete()
        super().delete(*args, **kwargs)

    
