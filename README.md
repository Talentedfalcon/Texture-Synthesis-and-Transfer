## Requirements to Run the Code:
- Import the “texture_synthesis.py” file to use all the different cut method as well as to use texture transfer.

## Introduction:
- The goal of this project is to implement the quilt cut method for texture synthesis (making a larger sized texture using a given smaller texture) and texture transfer (copy the certain attributes of a texture to another image) mentioned in [Efros and Freeman (2001)](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/papers/efros-siggraph01.pdf)

## Definitions:
- **Block/Patch**: A small section from the texture image
- **Synthesis Image**: The final output image generated after synthesis
- **Template**: Overlapped region of the synthesis image and block/patch
- **Minimum Error Boundary Cut**: The error between the template and patch is calculated using sum squared error and then using the error map we use Djikstra’s Algorithm to find the least error path along the longest axis of the error map

## Analysis:
1. ### Texture Synthesis:
    - Here are some methods I tried before implementing the actual quilt cut:
        - **Generic Block Tiling**:
            - Choose an initial patch and just keep placing them next to each other in synthesis image
        - **Random Block Tiling**:
            - Choose a random block then place then next to each other
        - **Random Block Tiling with Boundary Overlap**:
            - Choose a random block then place it next to each other with defined overlap in the x and y direction (if possible)
        - **Best Block Tiling with Boundary Overlap**:
            - Search block-by-block for a patch in the texture with the least error compared to the overlap region and then place it with defined overlap in the x and y direction (if possible)

    - #### Random Block Tiling with Minimum Error Boundary Cut:
        - **Example of Cut**:

            
        - **Output**:



    - #### Best Block Tiling with Minimum Error Boundary Cut:
        - **Example of Cut**:

            
        - **Output**:


    - #### Threshold Error Block Select with Boundary Overlap:
        - In this method the Error is computed using *Sum Squared Error of template and mask*, *Convolution of texture image and template*, *Convolution of texture image and mask*. With that we will get an error map for the whole texture image now we find valid points in texture image within an error threshold and choose such a point randomly.
            
        - **Example of Cut**:

            
        - **Output**:


    - #### Threshold Error Block Select with Minimum Error Boundary Cut:
        - **Example of Cut**:

            
        - **Output**:


2. ### Texture Transfer:
    - Here the *Threshold Error Block Select with Minimum Error Boundary Cut* method is used to get patches and stitch them. The output will be the same size as the target image
    - We achieve this by getting:
        - Error map of template and texture => ***error1***
        - Error map of target image patch and texture => ***error2***
    - Now we get ***(α\*error1 + (1-α)\*error2)***. Now we apply the minimum error boundary cut method used in the previously
        
    - **Example of Cut**:

        
    - **Output**:

