from django.urls import include, path
from . import views
from django.conf import settings
from django.conf.urls.static import static


# URL patterns for mapping HTTP requests to views.



urlpatterns = [
    path("", views.start, name="start"),   # Route to the 'start' view, accessed at '/'
    path("home/", views.home, name="home"),  # Route to the 'home' view, accessed at 'home/'
    path("search/", views.search, name="search"),  # Route to the 'search' view, accessed at 'search/'
    path("team/", views.equipe, name="equipe"),  # Route to the 'equipe' view, accessed at 'team/'
    path("help/histogram/", views.help_histogram, name="help_histogram"), # Route to the 'help_histogram' view, accessed at 'help/histogram/'
    path("help/image_retrieval/", views.image_retrieval, name="help_image_retrieval"),  # Route to the 'image_retrieval' view, accessed at 'help/image_retrieval/'
    path('upload/image/', views.upload_image_faiss, name='upload_image_faiss'),  # Route to the 'upload_image_faiss' view, accessed at 'upload/image/'
    path('upload/image/', views.upload_image, name='upload_image'),  # Route to the 'upload_image_faiss' view, accessed at 'upload/image/'
    path('refine/results/', views.refine_results, name='refine_results'), # Route to the 'refine_results' view, accessed at 'refine/results/'
    path('download/similar_images/', views.download_similar_images, name='download_similar_images'),  # Route to the 'download_similar_images' view, accessed at 'download/similar_images/'
    path('feature_maps/', views.show_feature_maps, name='feature_maps'),  # Route to the 'show_feature_maps' view, accessed at '/feature_maps/'
    path('style_analysis/', views.style_analysis, name='style_analysis'),  # Route to the 'style_analysis' view, accessed at '/style_analysis/'
    path('terms_of_use/', views.terms_of_use, name='terms_of_use'),  # Route to the 'terms_of_use' view, accessed at '/terms_of_use/
     

] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)


