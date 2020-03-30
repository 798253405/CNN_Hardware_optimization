import tensorflow as tf
import numpy as np



def conv2d_rewrite(input, filter,strides,padding):

    #input=batch*height*width*channel,\
    # for example: 128*32*32*1 means 128imgaes, size =32*32,1channels
    batch_size = input.shape[0]
    img_row=input.shape[1]
    img_col=input.shape[2]
    img_cha=input.shape[3]
    filter_row=filter.shape[0]#filter=height*widdth*channel*number_of_kernel
    filter_col = filter.shape[1]
    out_depth = filter.shape[3]

    filter_in_col=filter.reshape(1,-1,img_cha,out_depth)#3*3*1*depth->1*9*1*depth
    print(filter.shape, 'filter before')
    print(filter_in_col.shape,'filter shape after')
    if padding == 'SAME':
         output_size=[img_row,img_col,out_depth]
         #output_size = [((img_row-filter_row)//strides+1),((img_col-filter_col)//strides+1),out_depth]
    output = np.zeros(shape=(batch_size,output_size[0],output_size[1],output_size[2]))#in rgb, output=r+g+b. in here, simplyfied.
    outputr=np.zeros(shape=(batch_size,output_size[0],output_size[1],output_size[2]))
    for i in range(batch_size):# the i_th img
        for out_j in range(int(output_size[0])):#j row in output
            for out_k in range(int(output_size[1])):#k col in output
                for out_l in  range(int(output_size[2])):#the out_l_th output
                    if padding == 'SAME':
                        #add zeros
                        input_pad = np.pad(input[i,:,:,0], ((filter_row// 2, filter_row // 2), (filter_col // 2, filter_col // 2)),
                                                   'constant', constant_values=0)

                    #part of input image: the same size of filter(choose 3*3 part from a 32*32 image)
                    img_part=input_pad[out_j*strides:out_j*strides+filter_row,out_k*strides:out_k*strides+filter_row]


                    outputr[i, out_j, out_k, out_l] = np.dot(img_part.reshape([-1]), filter_in_col[:,:,:,out_l])
                   #for black*white, don't need to consider RGB
                  #  output[i, out_j, out_k, out_l] =  outputr[i, out_j, out_k, out_l] +\
                  #      outputg[i, out_j, out_k, out_l] + outputb[i, out_j, out_k, out_l]
                    output[i, out_j, out_k, out_l] = outputr[i, out_j, out_k, out_l]

    return output



#below for testing
input = tf.random.truncated_normal((1,32,32,1), stddev=0.1, dtype=tf.dtypes.float64)
filter = tf.random.truncated_normal((3,3,1,1), stddev=0.1, dtype=tf.dtypes.float64)


test= conv2d_rewrite(input,filter.numpy(),1,'SAME')
#print(test)

tf_test=tf.nn.conv2d(input,filter, strides=[1, 1, 1, 1], padding='SAME')

t_zero=tf_test-test
print(t_zero)
print(tf_test.shape,'tf test')
print(test.shape,'rewrite test')
