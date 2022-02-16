from skimage import io, color, img_as_ubyte
from skimage.feature import greycomatrix, greycoprops
from sklearn.metrics.cluster import entropy
def calc_entropy(image):
    red= image[:,:,0]
    blue= image[:,:,1]
    green= image[:,:,2]
    distances = [1, 2, 3]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    properties = ['energy', 'homogeneity'] #

    red_glcm = greycomatrix(red, distances=distances, angles=angles,symmetric=True, normed=True) 
    red_entropy= entropy(red_glcm)

    green_glcm = greycomatrix(green, distances=distances, angles=angles,symmetric=True, normed=True) 
    green_entropy= entropy(green_glcm)
    
    blue_glcm = greycomatrix(blue, distances=distances, angles=angles,symmetric=True, normed=True) 
    blue_entropy= entropy(blue_glcm)
    
    total_entropy= red_entropy+ green_entropy + blue_entropy
    
    return total_entropy
  
  ## Test
  image= cv2.imread(image_files[0])
  print(calc_entropy(image))
