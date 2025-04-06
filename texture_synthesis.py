import random
import numpy as np
import heapq
import matplotlib.pyplot as plt
import cv2 as cv

def random_num(a:int):
    '''
    Get a random integer between [0,a)
    '''
    return random.randint(0, a)


def get_random_patch(texture:np.ndarray,synthesis_img_size:tuple,block_size:tuple,y:int,x:int)->np.ndarray:
    '''
    Select a random patch from `texture` of size `block_size`-`(y,x)` 
    '''
    offset=(random_num(texture.shape[0]-block_size[0]),random_num(texture.shape[1]-block_size[1]))
    fill_size=list(block_size)
    if(y+block_size[0]>=synthesis_img_size[0]):
        fill_size[0]=synthesis_img_size[0]-y
    if(x+block_size[1]>=synthesis_img_size[1]):
        fill_size[1]=synthesis_img_size[1]-x
    patch=texture[offset[0]:offset[0]+fill_size[0],offset[1]:offset[1]+fill_size[1]]

    return patch

def get_best_patch(texture:np.ndarray, synthesis_img:np.ndarray, block_size:tuple, overlap:int, y:int, x:int)->np.ndarray:
    '''
    Select the patch with the least error with the overlapped area. (Least error search is done iteratively)
    '''
    fill_size=list(block_size)
    if(y+block_size[0]>=synthesis_img.shape[0]):
        fill_size[0]=synthesis_img.shape[0]-y
    if(x+block_size[1]>=synthesis_img.shape[1]):
        fill_size[1]=synthesis_img.shape[1]-x

    errors = np.zeros((texture.shape[0] - fill_size[0], texture.shape[1] - fill_size[1]))
    for i in range(texture.shape[0] - fill_size[0]):
        for j in range(texture.shape[1] - fill_size[1]):
            patch = texture[i:i+fill_size[0], j:j+fill_size[1]]
            e = overlap_error(patch, synthesis_img, fill_size, overlap, y, x)
            errors[i, j] = e

    i, j = np.unravel_index(np.argmin(errors), errors.shape)
    return texture[i:i+fill_size[0], j:j+fill_size[1]]

def overlap_error(patch:np.ndarray, synthesis_img:np.ndarray, block_size:tuple, overlap:int, y:int, x:int)->np.ndarray:
    '''
    Get overlap error of `patch` with the overlapped region of the `synthesis_img`
    '''
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

def min_cut_path(errors:np.ndarray)->np.ndarray:
    '''
    Djikstra's Algorithm in the vertical direction and the destination is any node at a depth h
    '''
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

def min_cut_patch(patch:np.ndarray,synthesis_img:np.ndarray,overlap:int,y:int,x:int,show_cut:bool=False)->np.ndarray:
    '''
    Apply the Minimum Boundary Cut to the patch along the overlapped region
    '''
    patch=patch.copy()
    min_cut=np.ones(patch.shape,dtype=bool)

    if(show_cut):
        fig,ax=plt.subplots(3,2)

    if x>0:
        left=patch[:,:overlap]-synthesis_img[y:y+patch.shape[0],x:x+overlap]
        leftL2=np.sum(left**2,axis=2)
        for i,j in enumerate(min_cut_path(leftL2)):
            min_cut[i,:j]=False
        if(show_cut):
            ax[0,0].imshow(leftL2)
            ax[0,1].imshow(min_cut[:,:,0],cmap="gray")
    if y>0:
        up=patch[:overlap,:]-synthesis_img[y:y+overlap,x:x+patch.shape[1]]
        upL2=np.sum(up**2,axis=2)
        for j,i in enumerate(min_cut_path(upL2.T)):
            min_cut[:i,j]=False
        if(show_cut):
            ax[1,0].imshow(upL2)
            ax[1,1].imshow(min_cut[:,:,0],cmap="gray")

    output=(patch*min_cut)+(synthesis_img[y:y+patch.shape[0],x:x+patch.shape[1]]*(~min_cut))
    if(show_cut):
        ax[2,0].imshow(patch)
        ax[2,1].imshow(patch*min_cut)
        plt.tight_layout()
        plt.show()
        plt.imshow(output)
        plt.show()

    return output

def primitive_texture_synthesis(texture:np.ndarray,synthesis_img_size:tuple,block_size:tuple,overlap:int,select_mode:str="random",fill_mode:str="default")->np.ndarray:
    '''
    Perform texture synthesis based on random block selection or best block selection

    Options:
    - `select_mode` = ['random' , 'best']
    - `fill_mode` = ['default' , 'min_boundary_cut']
    '''
    select_mode=select_mode.lower()
    fill_mode=fill_mode.lower()

    synthesis_img=np.zeros(synthesis_img_size)
    for y in range(0,synthesis_img.shape[0],block_size[0]-overlap):
        for x in range(0,synthesis_img.shape[1],block_size[1]-overlap):
            if(select_mode=="random"):
                patch=get_random_patch(texture,synthesis_img_size,block_size,y,x)
            elif(select_mode=="best"):
                patch=get_best_patch(texture,synthesis_img,block_size,overlap,y,x)
            else:
                raise Exception("Invalid select mode")
            
            if(fill_mode=="min_boundary_cut"):
                patch=min_cut_patch(patch,synthesis_img,overlap,y,x)
            elif(fill_mode!="default"):
                raise Exception("Invalid fill mode")
                
            synthesis_img[y:y+block_size[0],x:x+block_size[1]]=patch
    return synthesis_img.astype(np.uint8)

def ssd_patch(template:np.ndarray, mask:np.ndarray, texture:np.ndarray)->np.ndarray:
    '''
    Performes template matching with the overlapping region, computing the cost of sampling each patch, based on the sum of squared differences (SSD).
    '''
    template = template.astype(np.float64)
    mask = mask.astype(np.float64)
    texture = texture.astype(np.float64)
    
    def ssd(ch: int):
        return (
            ((mask[:,:,ch]*template[:,:,ch])**2).sum()\
            - 2 * cv.filter2D(texture[:,:,ch], ddepth=-1, kernel=template[:,:,ch]) \
            + cv.filter2D(texture[:,:,ch] ** 2, ddepth=-1, kernel=mask[:,:,ch])
        )

    ssd_b = ssd(0)
    ssd_g = ssd(1)
    ssd_r = ssd(2)

    return ssd_b + ssd_g + ssd_r

def choose_sample(cost:np.ndarray, tolerance:int)->tuple:
    '''
    Sort the first `tolerance` costs then randomly choose one of those costs x,y position
    '''
    idx = np.argpartition(cost.ravel(), tolerance-1)[:tolerance]
    lowest_cost = np.column_stack(np.unravel_index(idx, cost.shape))
    return random.choice(lowest_cost)

def quilt_simple(texture:np.ndarray,synthesis_img_size:tuple,block_size:tuple,overlap:int,tolerance:int,show_cut:bool=False)->np.ndarray:
    '''
    Selects the next patch by template matching the texture with the overlapping region. (NO min_boundary_cut operation is performed)
    '''
    synthesis_img = np.zeros(synthesis_img_size)
    
    offset = (block_size[0] - overlap, block_size[1]-overlap)
    
    for y in range(0, synthesis_img_size[0], offset[0]):
        for x in range(0, synthesis_img_size[1], offset[1]):
            fill_size=list(block_size)
            if(y+block_size[0]>=synthesis_img_size[0]):
                fill_size[0]=synthesis_img_size[0]-y
            if(x+block_size[1]>=synthesis_img_size[1]):
                fill_size[1]=synthesis_img_size[1]-x

            template = synthesis_img[y:y+fill_size[0], x:x+fill_size[1], :].copy()
            mask = np.zeros(fill_size)
            
            if y == 0:
                mask[:, :overlap, :] = 1
            elif x == 0:
                mask[:overlap, :, :] = 1
            else:
                mask[:, :overlap, :] = 1
                mask[:overlap, :, :] = 1
                
            half = [fill_size[0]//2, fill_size[1]//2]
                
            ssd = ssd_patch(template, mask, texture)
            if(show_cut):
                plt.imshow(ssd)
                plt.show()
            
                temp=ssd.copy()
                temp[:half[0],:]=0
                temp[:,:half[1]]=0
                temp[-half[0]:,:]=0
                temp[:,-half[1]:]=0
                plt.imshow(temp)
                plt.show()
            
            ssd = ssd[half[0]:-half[0], half[1]:-half[1]]
            i, j = choose_sample(ssd, tolerance)
            synthesis_img[y:y+fill_size[0], x:x+fill_size[1], :] = texture[i:i+fill_size[0], j:j+fill_size[1], :]
            
            if(show_cut):
                plt.imshow(template.astype(np.uint8))
                plt.show()
                plt.imshow(mask)
                plt.show()
                plt.imshow(texture[i:i+fill_size[0], j:j+fill_size[1], :])
                plt.show()
    return synthesis_img.astype(np.uint8)

def mask_cut(error:np.ndarray)->np.ndarray:
    '''
    Cut a mask vertically based on the path defined from `min_cut_path` function
    '''
    mask_patch=np.zeros((error.shape[0],error.shape[1],3))
    for i,j in enumerate(min_cut_path(error)):
        mask_patch[i,:j]=1
    return mask_patch

def quilt_cut(texture:np.ndarray,synthesis_img_size:tuple,block_size:tuple,overlap:int,tolerance:int,show_cut:bool=False)->np.ndarray:
    '''
    Selects the next patch by template matching the texture with the overlapping region and using min_boundary_cut
    '''
    synthesis_img = np.zeros(synthesis_img_size)
    offset = (block_size[0] - overlap, block_size[1]-overlap)
    
    for y in range(0, synthesis_img_size[0], offset[0]):
        for x in range(0, synthesis_img_size[1], offset[1]):
            fill_size=list(block_size)
            if(y+block_size[0]>=synthesis_img_size[0]):
                fill_size[0]=synthesis_img_size[0]-y
            if(x+block_size[1]>=synthesis_img_size[1]):
                fill_size[1]=synthesis_img_size[1]-x

            if x+y==0:
                synthesis_img[y:y+block_size[0], x:x+block_size[1], :] = texture[0:block_size[0], 0:block_size[1], :].copy()
                continue

            template = synthesis_img[y:y+fill_size[0], x:x+fill_size[1], :].copy().astype(np.uint8)
            mask = np.zeros(fill_size)
            
            if y == 0:
                mask[:, :overlap, :] = 1
            elif x == 0:
                mask[:overlap, :, :] = 1
            else:
                mask[:, :overlap, :] = 1
                mask[:overlap, :, :] = 1
                
            half = [fill_size[0]//2, fill_size[1]//2]
                
            if(show_cut):
                fig,axes=plt.subplots(2,7,figsize=(18,4),facecolor='lightgray')
                for ax in axes.flat:
                    ax.axis('off')

            ssd = ssd_patch(template, mask, texture)
            if(show_cut):
                axes[0,0].imshow(ssd)
                axes[0,0].set_title("Error of Texture with Template")
            
                temp=ssd.copy()
                temp[:half[0],:]=0
                temp[:,:half[1]]=0
                temp[-half[0]:,:]=0
                temp[:,-half[1]:]=0
                axes[1,0].imshow(temp)
                axes[1,0].set_title("Clipped Error")
            
            ssd = ssd[half[0]:-half[0], half[1]:-half[1]]
            i, j = choose_sample(ssd, tolerance)
            
            patch = texture[i:i+fill_size[0], j:j+fill_size[1], :].copy()
                
            mask1 = np.zeros(fill_size)
            if(y!=0):
                diff1 = (template[:overlap, :fill_size[0], :] - patch[:overlap, :fill_size[0], :]) ** 2
                diff1 = np.sum(diff1, axis=2)
                if(show_cut):
                    axes[0,1].imshow(diff1.astype(np.uint8))
                    axes[0,1].set_title("Horizontal Overlap Error")
                mask_patch1 = mask_cut(diff1.T).transpose(1,0,2)
                mask1[:overlap, :fill_size[0]] = mask_patch1
            if(show_cut):
                axes[0,2].imshow(mask1)
                axes[0,2].set_title("Horizontal Overlap Mask")

            mask2 = np.zeros(fill_size)
            if(x!=0):
                diff2 = (template[:fill_size[1], :overlap, :] - patch[:fill_size[1], :overlap, :]) ** 2
                diff2 = np.sum(diff2, axis=2)
                if(show_cut):
                    axes[1,1].imshow(diff2.astype(np.uint8))
                    axes[1,1].set_title("Vertical Overlap Error")
                mask_patch2 = mask_cut(diff2)
                mask2[:fill_size[1], :overlap] = mask_patch2
            if(show_cut):
                axes[1,2].imshow(mask2)
                axes[1,2].set_title("Vertical Overlap Mask")

            if(x==0):
                full_mask_patch=mask1.astype(np.uint8)
            elif(y==0):
                full_mask_patch=mask2.astype(np.uint8)
            else:
                full_mask_patch=np.logical_or(mask1,mask2).astype(np.uint8)
            if(show_cut):
                ax5=plt.subplot(1,7,4)
                ax5.imshow(full_mask_patch.astype(np.float32))
                ax5.set_title("Combined Mask")
                ax5.axis('off')
                axes[0,4].imshow(template)
                axes[0,4].set_title("Template Full")
                axes[1,4].imshow(patch)
                axes[1,4].set_title("Patch Full")

            template = (template*full_mask_patch).astype(np.uint8)
            full_mask_patch^=1
            patch = (patch*full_mask_patch).astype(np.uint8)
            
            if(show_cut):
                axes[0,5].imshow(template)
                axes[0,5].set_title("Template Masked")
                axes[1,5].imshow(patch)
                axes[1,5].set_title("Patch Masked")
                ax6=plt.subplot(1,7,7)
                ax6.imshow(patch+template)
                ax6.set_title("Combined Patch")
                ax6.axis('off')

                plt.tight_layout()
                plt.show()

            synthesis_img[y:y+fill_size[0], x:x+fill_size[1], :] = patch+template       
    return synthesis_img.astype(np.uint8)

def texture_transfer(texture:np.ndarray,target:np.ndarray,block_size:tuple,overlap:int,tolerance:int,alpha:float,show_cut:bool=False)->np.ndarray:
    '''
    Reconstruct `target` image using `texture` using texture synthesis principles
    '''
    offset = (block_size[0] - overlap, block_size[1]-overlap)
    
    transfered=np.zeros(target.shape)

    for y in range(0, target.shape[0], offset[0]):
        for x in range(0, target.shape[1], offset[1]):
            fill_size=list(block_size)
            if(y+block_size[0]>=target.shape[0]):
                fill_size[0]=target.shape[0]-y
            if(x+block_size[1]>=target.shape[1]):
                fill_size[1]=target.shape[1]-x

            template = transfered[y:y+fill_size[0], x:x+fill_size[1], :].copy().astype(np.uint8)
            _target=target[y:y+fill_size[0], x:x+fill_size[1], :].copy()
            mask = np.zeros(fill_size)
            
            if y == 0:
                mask[:, :overlap, :] = 1
            elif x == 0:
                mask[:overlap, :, :] = 1
            else:
                mask[:, :overlap, :] = 1
                mask[:overlap, :, :] = 1
                
            half = [fill_size[0]//2, fill_size[1]//2]
                
            if(show_cut):
                fig,axes=plt.subplots(2,8,figsize=(18,4),facecolor='lightgray')
                for ax in axes.flat:
                    ax.axis('off')

            ssd_overlap = ssd_patch(template, mask, texture)
            ssd_target=ssd_patch(_target,np.ones((fill_size[0], fill_size[1], 3)),texture)
            if(show_cut):
                axes[0,0].imshow(ssd_overlap)
                axes[0,0].set_title("Error of Texture with Template")
            
                temp=ssd_overlap.copy()
                temp[:half[0],:]=0
                temp[:,:half[1]]=0
                temp[-half[0]:,:]=0
                temp[:,-half[1]:]=0
                axes[1,0].imshow(temp)
                axes[1,0].set_title("Clipped Error")

                axes[0,1].imshow(ssd_overlap)
                axes[0,1].set_title("Error of Texture with Target")
            
                temp=ssd_overlap.copy()
                temp[:half[0],:]=0
                temp[:,:half[1]]=0
                temp[-half[0]:,:]=0
                temp[:,-half[1]:]=0
                axes[1,1].imshow(temp)
                axes[1,1].set_title("Clipped Error")
            
            ssd_overlap = ssd_overlap[half[0]:-half[0], half[1]:-half[1]]
            ssd_target = ssd_target[half[0]:-half[0], half[1]:-half[1]]

            ssd=(alpha*ssd_overlap)+((1-alpha)*ssd_target)

            i, j = choose_sample(ssd, tolerance)
            
            patch = texture[i:i+fill_size[0], j:j+fill_size[1], :].copy()
                
            mask1 = np.zeros(fill_size)
            if(y!=0):
                diff1 = (template[:overlap, :fill_size[0], :] - patch[:overlap, :fill_size[0], :]) ** 2
                diff1 = np.sum(diff1, axis=2)
                if(show_cut):
                    axes[0,2].imshow(diff1.astype(np.uint8))
                    axes[0,2].set_title("Horizontal Overlap Error")
                mask_patch1 = mask_cut(diff1.T).transpose(1,0,2)
                mask1[:overlap, :fill_size[0]] = mask_patch1
            if(show_cut):
                axes[0,3].imshow(mask1)
                axes[0,3].set_title("Horizontal Overlap Mask")

            mask2 = np.zeros(fill_size)
            if(x!=0):
                diff2 = (template[:fill_size[1], :overlap, :] - patch[:fill_size[1], :overlap, :]) ** 2
                diff2 = np.sum(diff2, axis=2)
                if(show_cut):
                    axes[1,2].imshow(diff2.astype(np.uint8))
                    axes[1,2].set_title("Vertical Overlap Error")
                mask_patch2 = mask_cut(diff2)
                mask2[:fill_size[1], :overlap] = mask_patch2
            if(show_cut):
                axes[1,3].imshow(mask2)
                axes[1,3].set_title("Vertical Overlap Mask")

            if(x==0):
                full_mask_patch=mask1.astype(np.uint8)
            elif(y==0):
                full_mask_patch=mask2.astype(np.uint8)
            else:
                full_mask_patch=np.logical_or(mask1,mask2).astype(np.uint8)
            if(show_cut):
                ax5=plt.subplot(1,8,5)
                ax5.imshow(full_mask_patch.astype(np.float32))
                ax5.set_title("Combined Mask")
                ax5.axis('off')
                axes[0,5].imshow(template)
                axes[0,5].set_title("Template Full")
                axes[1,5].imshow(patch)
                axes[1,5].set_title("Patch Full")

            template = (template*full_mask_patch).astype(np.uint8)
            full_mask_patch^=1
            patch = (patch*full_mask_patch).astype(np.uint8)
            
            if(show_cut):
                axes[0,6].imshow(template)
                axes[0,6].set_title("Template Masked")
                axes[1,6].imshow(patch)
                axes[1,6].set_title("Patch Masked")
                ax6=plt.subplot(1,8,8)
                ax6.imshow(patch+template)
                ax6.set_title("Combined Patch")
                ax6.axis('off')

                plt.tight_layout()
                plt.show()

            transfered[y:y+fill_size[0], x:x+fill_size[1], :] = patch+template 
    return transfered.astype(np.uint8)