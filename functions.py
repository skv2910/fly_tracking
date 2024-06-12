def keys_in_image_resolution(image,keypoints):   #retusns keys in format compatible with image resolution[(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
    img_height,img_width=image.shape
    out=[]
    for key in keypoints:
        x_n,y_n,=key
        x=int(x_n*img_width)
        y=int(y_n*img_height)
        key_=(x,y)
        out.append(key_)
    return out
def box_in_image_resolution(image,box):  #return box compatible with image resolution [x,y,w,h]
    img_height,img_width=image.shape
    
    x_center_norm, y_center_norm, width_norm, height_norm = box
    x=x_center_norm*img_width
    y=y_center_norm*img_height
    w=width_norm*img_width
    h=height_norm*img_height
    out=[x,y,w,h]
    return out






def draw_annotations(image,bbox, keypoints):    #for drawing annotations on image
    # Draw bounding boxes
    img_height, img_width = image.shape
    x_center_norm, y_center_norm, width_norm, height_norm = bbox
    x_center = int(x_center_norm * img_width)
    y_center = int(y_center_norm * img_height)
    width = int(width_norm * img_width)
    height = int(height_norm * img_height)
    x1 = int(x_center - width / 2)-20
    y1 = int(y_center - height / 2)-20
    x2 = int(x_center + width / 2)+20
    y2 = int(y_center + height / 2)+20
        
    color = (0, 255, 0)  # Green color in BGR
    thickness = 2

# Draw rectangle
    cv.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    for keypoint in keypoints:
        x_normalized, y_normalized = keypoint
        x = int(x_normalized * image.shape[1])
        y = int(y_normalized * image.shape[0])
        
        cv.circle(image, (x, y), 4, (0, 0, 255), -1)
    return image










