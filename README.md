Change detection in an aerial video feed – A guide on installation and execution of the project
This project utilizes a Siamese CNN to effectively perform change detection action on multi temporal video feeds.
Note: Cuda is necessary to run the siamese CNN model
1.	Installation Instructions:
    •	If an environment is preferred, run the following command in anaconda prompt:

    o	Create an environment:
    conda create --name env_name
    o	To Activate the environment:
     conda activate env_name        	
    •	Download and Install Python version greater than 3.7
    •	Install pip
    •	Download and Install any IDE
    •	Pytorch installation
    o	conda install pytorch -c pytorch
    o	pip install torchvision

    •	Django installation
    o	pip install django
    o	After successful installation to check the version of Django
    django-admin --version

    •	Other python modules:
    o	pip install matplotlib
    o	pip install numpy
    o	pip install Pillow
    o	pip install tqdm
    o	pip install opencv-python

2.	Download the project repo, unzip into your system.
    Navigate to the project folder using the command:
    cd path_to_folder/mysite
3.	On navigating to mysite run the server by using
    python manage.py runserver
    This command runs the project in your localhost and the port number is 8000. You can check the output on your web browser. You have to run the above command every time the server is down.
    The app is called changedetection and encompasses several files which are elaborated below:
    The changedetection directory consists of three python files.
    1.	urls.py
    2.	form.py
    3.	code.py
    urls.py is the file that contains the url for main page
    form.py is the file that contains the code for input form on webpage
    code.py is the file where all the functions and operations are defined.
    Within the folder mysite (root folder) there exists 3 folders namely, media, static and template.
    media folder stores the video, frames, panoramas and final output.
    static folder contains css files, images and js file
    template file contains all the html files.

4.	On the web page:
    After the result page is displayed on the webpage, ensure to click on 'list' button and delete the videos uploaded. This prevents the previously uploaded videos from being displayed on the upload page. 
    Additionally, also click on the ‘delete all’ button to remove all the video related files (videos input, frames, panorama) saved locally.


