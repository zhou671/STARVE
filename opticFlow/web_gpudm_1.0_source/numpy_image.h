/*
Copyright (C) 2015 Jerome Revaud

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>
*/
#ifndef ___numpy_image___
#define ___numpy_image___

/************************
* 1D Array
*/

#define DEFINE_ARRAY(type)  \
    typedef struct {  \
      type* pixels; \
      int tx; \
    } type##_array;

DEFINE_ARRAY(int)
DEFINE_ARRAY(float)

#define ASSERT_ARRAY_ZEROS(arr) {int size=arr->tx; assert((arr->pixels[0]==0 && arr->pixels[size/2]==0 && arr->pixels[size-1]==0) || !"error: matrix " #arr "is supposed to be zeros");}
#define ARRAY_SIZE(array)   (array->tx)
#define ASSERT_SAME_ARRAY_SIZE(arr1,arr2) assert( (arr1)->tx == (arr2)->tx )

/************************
* 2D Image
*/

#define DEFINE_IMG(type)    \
    typedef struct { \
      type* pixels;\
      int tx,ty;\
    } type##_image;

DEFINE_IMG(int)
DEFINE_IMG(float)

#define ASSERT_SAME_SIZE  ASSERT_SAME_IMG_SIZE
#define ASSERT_IMG_SIZE  ASSERT_SAME_IMG_SIZE
#define ASSERT_SAME_IMG_SIZE(im1,im2)  if(im1 && im2)  assert(im1->tx==im2->tx && im1->ty==im2->ty);

#define ASSERT_IMAGE_ZEROS
#define ASSERT_IMG_ZEROS(img) {int size=img->tx*img->ty; assert((img->pixels[0]==0 && img->pixels[size/2]==0 && img->pixels[size-1]==0) || !"error: matrix " #img "is supposed to be zeros");}
#define IMG_SIZE(cube) ((cube)->tx*(cube)->ty)


/************************
* 3D Image = Cube (Z coordinates are contiguous)
*/

#define DEFINE_CUBE(type) \
    typedef struct {  \
      type* pixels;  \
      int tx,ty,tz;  \
    } type##_cube;

DEFINE_CUBE(int)
DEFINE_CUBE(float)

#define ASSERT_SAME_CUBE_SIZE(im1, im2)   \
  if((im1) && (im2))  assert((im1)->tx==(im2)->tx && (im1)->ty==(im2)->ty && (im1)->tz==(im2)->tz);

#define ASSERT_CUBE_ZEROS(img) {int size=img->tx*img->ty*img->tz; assert((img->pixels[0]==0 && img->pixels[size/2]==0 && img->pixels[size-1]==0) || !"error: matrix " #img "is supposed to be zeros");}
#define CUBE_SIZE(cube) ((cube)->tx*(cube)->ty*(cube)->tz)

#endif


































