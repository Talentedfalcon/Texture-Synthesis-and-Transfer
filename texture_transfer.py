import numpy as np
import heapq

def get_random_patch(img,synthesis_img_size,block_size,y,x):
    offset=(np.random.randint(0,img.shape[0]-block_size[0]),np.random.randint(0,img.shape[1]-block_size[1]))
    fill_size=list(block_size)
    if(y+block_size[0]>=synthesis_img_size[0]):
        fill_size[0]=synthesis_img_size[0]-y
    if(x+block_size[1]>=synthesis_img_size[1]):
        fill_size[1]=synthesis_img_size[1]-x
    patch=np.astype(img[offset[0]:offset[0]+fill_size[0],offset[1]:offset[1]+fill_size[1]],np.uint8)

    return patch

#Djikstra's Algorithm in the vertical direction and the destination is any node at a depth h
def min_cut_path(errors):
    pq=[(error,[i]) for i,error in enumerate(errors[0])]
    heapq.heapify(pq)
    h,w=errors.shape
    visited=set()
    while pq:
        error,path=heapq.heappop(pq)
        current_depth=len(path)
        current_index=path[-1]
        if current_depth==h:
            return path
        for i in -1,0,1:
            next_index=current_index+i
            if 0<=next_index<w:
                if(current_depth,next_index) not in visited:
                    cumError=error+errors[current_depth,next_index]
                    heapq.heappush(pq,(cumError,path+[next_index]))
                    visited.add((current_depth,next_index))

def min_cut_patch(patch,overlap,synthesis_img,y,x):
    patch=patch.copy()
    min_cut=np.zeros_like(patch,dtype=bool)

    if x>0:
        left=patch[:,:overlap]-synthesis_img[y:y+patch.shape[0],x:x+overlap]
        leftL2=np.sum(left**2,axis=2)
        for i,j in enumerate(min_cut_path(leftL2)):
            min_cut[i,:j]=True
    if y>0:
        up=patch[:overlap,:]-synthesis_img[y:y+overlap,x:x+patch.shape[1]]
        upL2=np.sum(up**2,axis=2)
        for j,i in enumerate(min_cut_path(upL2.T)):
            min_cut[:i,j]=True

    np.copyto(patch, synthesis_img[y:y+patch.shape[0], x:x+patch.shape[1]], where=min_cut)

    return patch

def overlap_error(patch, synthesis_img, block_size, overlap, y, x):
    error = 0
    if x > 0:
        left = patch[:, :overlap] - synthesis_img[y:y+block_size[0], x:x+overlap]
        error += np.sum(left**2)
    if y > 0:
        up   = patch[:overlap, :] - synthesis_img[y:y+overlap, x:x+block_size[1]]
        error += np.sum(up**2)
    if x > 0 and y > 0:
        corner = patch[:overlap, :overlap] - synthesis_img[y:y+overlap, x:x+overlap]
        error -= np.sum(corner**2)
    return error

def get_best_patch(img, synthesis_img, block_size, overlap, y, x):
    fill_size=list(block_size)
    if(y+block_size[0]>=synthesis_img.shape[0]):
        fill_size[0]=synthesis_img.shape[0]-y
    if(x+block_size[1]>=synthesis_img.shape[1]):
        fill_size[1]=synthesis_img.shape[1]-x

    errors = np.zeros((img.shape[0] - fill_size[0], img.shape[1] - fill_size[1]))
    for i in range(img.shape[0] - fill_size[0]):
        for j in range(img.shape[1] - fill_size[1]):
            patch = img[i:i+fill_size[0], j:j+fill_size[1]]
            e = overlap_error(patch, synthesis_img, fill_size, overlap, y, x)
            errors[i, j] = e

    i, j = np.unravel_index(np.argmin(errors), errors.shape)
    return img[i:i+fill_size[0], j:j+fill_size[1]]

def synthesis_texture(img,synthesis_img_size,block_size,overlap=0,patch_select_mode="random",fill_mode="tile"):
    if(overlap>block_size[0] or overlap>block_size[1]):
        raise Exception ("Invalid overlap size. (Overlap always < block_size)")
    patch_select_mode=patch_select_mode.lower()
    fill_mode=fill_mode.lower()

    if(patch_select_mode=="best" and fill_mode=="tile"):
        fill_mode="overlap"
        if(not overlap):
            raise Exception("'best' patch select requires overlap>0")
    
    synthesis_img=np.zeros(synthesis_img_size).astype(np.uint8)

    for y in range(0,synthesis_img.shape[0],block_size[0]-overlap):
        for x in range(0,synthesis_img.shape[1],block_size[1]-overlap):
            if(patch_select_mode=="random"):
                patch=get_random_patch(img,synthesis_img_size,block_size,y,x)
            elif(patch_select_mode=="best"):
                patch=get_best_patch(img,synthesis_img,block_size,overlap,y,x)
            else:
                raise Exception ("Invalid patch select mode")
    
            if(fill_mode=="min_bound_cut"):
                patch=min_cut_patch(patch,overlap,synthesis_img,y,x)
            if(fill_mode=="tile" or fill_mode=="overlap" or fill_mode=="min_bound_cut"):
                synthesis_img[y:y+block_size[0],x:x+block_size[1]]=patch
            else:
                raise Exception("Invalid fill mode")
            
    return synthesis_img