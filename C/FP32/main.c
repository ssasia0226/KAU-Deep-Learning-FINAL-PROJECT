#define _CRT_SECURE_NO_WARNINGS
#define data_size 48600
#define input_data_width 32
#define input_data_ch 1
//input data(input_data_width, input_data_width)
#define conv1_in_width input_data_width
#define conv1_in_ch input_data_ch
#define conv1_out_width 24
#define conv1_out_ch 256
#define conv1_stride 1
#define conv2_in_width conv1_out_width
#define conv2_in_ch conv1_out_ch
#define conv2_out_ch 256
#define conv2_out_width 8
#define conv2_stride 2
#define kernel_width 9
#define num_primary_caps 2048
#define dim_primary_caps 8
//primarycaps(dim_primary_caps, num_primary_caps)
#define dim_predic_vector 16
#define num_class 5
//prediction_vectors(num_primary_caps, num_class, dim_predic_vector)
#define num_iterations 1

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
float conv1_kernel[conv1_out_ch * conv1_in_ch * kernel_width * kernel_width];           // ch ���� Ȯ���غ�����
float conv2_kernel[conv2_out_ch * conv2_in_ch * kernel_width * kernel_width];
float conv1_output[conv1_out_ch * conv1_out_width * conv1_out_width];
float conv2_output[conv2_out_ch * conv2_out_width * conv2_out_width];
float conv1_bias[conv1_out_ch];
float conv2_bias[conv2_out_ch];
float digits_W[num_primary_caps * num_class * dim_predic_vector * dim_primary_caps];
float digits_bias[num_class * dim_predic_vector];
float u_hat[num_primary_caps * num_class * dim_predic_vector];
float result_v[num_class * dim_predic_vector];

void convolution(float* input, float* kernel, float* bias, float* output, int input_width,
    int input_ch, int output_width, int output_ch, int stride);
void squash(float* input, float* output, int dim_caps, int num_caps);
void ReLU(float* input);
void prediction_vectors(float* input, float* weight_matrix, float* output);
void softmax(float* input, float* output);
void dynamic_routing(float* uhat, float* bias, float* v_j);


void convolution(float* input, float* kernel, float* bias, float* output, int input_width,
    int input_ch, int output_width, int output_ch, int stride) {
    int i, j, k, p, l, m;
    for (i = 0; i < output_width; i++) {
        for (j = 0; j < output_width; j++) {
            for (k = 0; k < output_ch; k++) {
                output[k * output_width * output_width + i * output_width + j] = bias[k];
                //output[k * output_width * output_width + i * output_width + j] = 0;
                for (p = 0; p < input_ch; p++) {
                    for (l = 0; l < kernel_width; l++) {
                        for (m = 0; m < kernel_width; m++) {
                            output[k * output_width * output_width + i * output_width + j] +=
                                input[input_width * input_width * p + input_width * (i * stride + l) + (j * stride + m)] *
                                kernel[k * kernel_width * kernel_width * input_ch + p * kernel_width * kernel_width + l * kernel_width + m];
                        }
                    }
                }
            }
        }
    }
}
void squash(float* input, float* output, int dim_caps, int num_caps) { //원래 squash func
    float tmp;
    for (int i=0; i < dim_caps; i++) {
        tmp = 0;
        for (int j=0; j < num_caps; j++) {
            tmp += pow(input[i + j * dim_caps], 2);
        }
        for (int k=0; k < num_caps; k++) {
            output[i + k * dim_caps] = (input[i + k * dim_caps] * sqrt(tmp)) / (tmp + 1);
        }
    }
}

void squash__(float* input, float* output, int dim_caps, int num_caps) {
    float tmp;
    for (int i=0; i < num_caps; i++) {
        tmp = 0;
        for (int j=0; j < dim_caps; j++) {
            tmp += pow(input[i * dim_caps + j], 2);
        }
        for (int k=0; k < dim_caps; k++) {
            output[i * dim_caps + k] = (input[i * dim_caps + k] * sqrt(tmp)) / (tmp + 1);
        }
    }
}

/*int log2(float argument){
    int arg = (int)argument;
    int exponent = 0;
    int argg=1;
    while(arg>1){
        arg>>=1;
        exponent++;
        argg<<=1;
    }
    if(((argg+(argg<<1))>>1)<=argument){
        exponent++;
    }
    return exponent;
}

void squash_approx(float* input, float* output, int dim_caps, int num_caps) {
    float tmp; int L2;
    for (int i=0; i < dim_caps; i++) {
        tmp = 0;
        for (int j=0; j < num_caps; j++) {
            tmp += input[i + j * dim_caps];
        }
        L2 = 1<<(log2(tmp)-1);
        for (int k=0; k < num_caps; k++) {
            output[i + k * dim_caps] = input[i + k * dim_caps] / (float)L2;
        }
    }
}*/
void ReLU(float* input) {
    for (int i=0; i < conv1_out_ch * conv1_out_width * conv1_out_width; i++) {
        if (input[i] < 0) input[i] = 0;
    }
}

void prediction_vectors(float* input, float* weight_matrix, float* output) {//1152 8 -> 1152 10 16
    for (int l = 0; l < num_primary_caps; l++) {
        for (int k = 0; k < num_class; k++) {
            for (int j = 0; j < dim_predic_vector; j++) {
                output[j + dim_predic_vector * k + dim_predic_vector * num_class * l] = 0;
                for (int i = 0; i < dim_primary_caps; i++) {
                    output[j + dim_predic_vector * k + dim_predic_vector * num_class * l] +=
                    weight_matrix[l * dim_predic_vector * dim_primary_caps * num_class + dim_predic_vector * dim_primary_caps * k + dim_primary_caps * j + i] *
                    input[i + l * dim_primary_caps];//transpose 안했으면 input[num_primary_caps * i + l];
                }
            }
        }
    }
}
void softmax(float* input, float* output) {
    float tmp;
    for (int i = 0; i < num_class; i++) {
        tmp = 0;
        for (int j = 0;j < num_primary_caps; j++) {
            tmp += exp(input[j * num_class + i]);
        }
        for (int k = 0;k < num_primary_caps; k++) {
            output[k * num_class + i] = exp(input[k * num_class + i]) / tmp;
        }
    }
}
void softmax_(float* input, float* output) {
    float tmp;
    for (int i = 0; i < num_primary_caps; i++) {
        tmp = 0;
        for (int j = 0;j < num_class; j++) {
            tmp += exp(input[j + i * num_class]);
        }
        for (int k = 0;k < num_class; k++) {
            output[k + i * num_class] = exp(input[k + i * num_class]) / tmp;
        }
    }
}
void dynamic_routing(float* uhat, float *bias, float* v_j) {
    float c_ij[num_primary_caps * num_class];
    float b_ij[num_primary_caps * num_class];
    float s_j[num_class * dim_predic_vector];
    //float u_vj1[num_primary_caps * num_class];
    float tmp;
    int i, j, k, l;
    for (i = 0;i < num_primary_caps * num_class;i++) {
        b_ij[i] = 0;
    }
    for (i = 0;i < num_iterations;i++) {
        softmax_(b_ij, c_ij);
        for (j = 0;j < num_class;j++) {
            for (k = 0;k < dim_predic_vector;k++) {
                tmp = 0;
                for (l = 0;l < num_primary_caps;l++) {
                    tmp += c_ij[num_class * l + j] * uhat[num_class * dim_predic_vector * l + dim_predic_vector * j + k];
                    //tmp += uhat[num_class * dim_predic_vector * l + dim_predic_vector * j + k];
                }
                s_j[j * dim_predic_vector + k] = tmp + bias[j * dim_predic_vector + k];
                //printf(" sj: %f ",s_j[k]);
            }
        }
        squash__(s_j, v_j, dim_predic_vector, num_class);
        for (j = 0; j < num_primary_caps; j++) {
            for (k = 0; k < num_class; k++) {
                tmp = 0;
                for (l = 0; l < dim_predic_vector; l++) {
                    tmp += uhat[j * num_class * dim_predic_vector + k * dim_predic_vector + l] * v_j[k * dim_predic_vector + l];
                    //tmp=u_vj1[num_class * j + k]
                }
                b_ij[num_class * j + k] += tmp;
            }
        } 
    } 
}
int main() {
    FILE* fp, * num_wrong;
    int predict = 0;
    float max = 0;
    float accuracy;
    float predict_sum;
    int correct = 0;
    int tmp_predict;
    int wrong = 0;
    float dataset[input_data_ch * input_data_width * input_data_width];
    float LABEL[data_size];

    fp = fopen("label_smallNORB_float.bin", "rb"); if (fp == NULL)
    {
        printf("Cannot open file.\n");
        exit(1);
    }
    fread(LABEL, sizeof(float), data_size, fp);
    printf("label : OK\n");
    printf("\n");
    ///////////////////////////////////////////

    fp = fopen("conv1_kernel_smallNORB_float.bin", "rb"); if (fp == NULL)
    {
        printf("Cannot open file.\n");
        exit(1);
    }
    fread(conv1_kernel, sizeof(float), conv1_out_ch * conv1_in_ch * kernel_width * kernel_width, fp);
    printf("conv1_kernel : OK\n");
    printf("\n");
    ///////////////////////////////////////////

    fp = fopen("conv1_bias_smallNORB_float.bin", "rb"); if (fp == NULL)
    {
        printf("Cannot open file.\n");
        exit(1);
    }
    fread(conv1_bias, sizeof(float), conv1_out_ch, fp);
    printf("conv1_bias : OK\n");
    printf("\n");
    ///////////////////////////////////////////

    fp = fopen("conv2_kernel_smallNORB_float.bin", "rb"); if (fp == NULL)
    {
        printf("Cannot open file.\n");
        exit(1);
    }
    fread(conv2_kernel, sizeof(float), conv2_out_ch * conv2_in_ch * kernel_width * kernel_width, fp);
    printf("conv2_kernel : OK\n");
    printf("\n");
    ///////////////////////////////////////////

    fp = fopen("conv2_bias_smallNORB_float.bin", "rb"); if (fp == NULL)
    {
        printf("Cannot open file.\n");
        exit(1);
    }
    fread(conv2_bias, sizeof(float), conv2_out_ch, fp);
    printf("conv1_bias : OK\n");
    printf("\n");
    ///////////////////////////////////////////

    fp = fopen("digit_caps_W_float.bin", "rb"); if (fp == NULL)
    {
        printf("Cannot open file.\n");
        exit(1);
    }
    fread(digits_W, sizeof(float), num_primary_caps * num_class * dim_predic_vector * dim_primary_caps, fp);
    printf("digits_W : OK\n");
    printf("\n");
    ///////////////////////////////////////////
    fp = fopen("digit_caps_bias_float.bin", "rb"); if (fp == NULL)
    {
        printf("Cannot open file.\n");
        exit(1);
    }
    fread(digits_bias, sizeof(float), num_class * dim_predic_vector, fp);
    printf("digits_bias : OK\n");
    printf("\n");
    ///////////////////////////////////////////

    fp = fopen("dataset_smallNORB_float.bin", "rb"); if (fp == NULL)
    {
        printf("Cannot open file.\n");
        exit(1);
    }

    for (int k = 0; k < 48600; k++) {
        fread(dataset, sizeof(float), input_data_ch * input_data_width * input_data_width, fp);
        convolution(dataset, conv1_kernel, conv1_bias, conv1_output, conv1_in_width, conv1_in_ch, conv1_out_width, conv1_out_ch, conv1_stride);
        ReLU(conv1_output);
        int notzero=0;
        //printf("conv1 layer completed.\n");
        ///////////////////////////////////////////
        /*for (int i = 0;i < 20;i++) {
            printf(" %f ", conv2_kernel[i]);
        }*/
        convolution(conv1_output, conv2_kernel, conv2_bias, conv2_output, conv2_in_width, conv2_in_ch, conv2_out_width, conv2_out_ch, conv2_stride);
        for (int i = 0; i < 1000; i++) {
            //printf(" %f ", conv2_output[i]);
        }
        
        //printf("\nconv2 layer completed.\n");
        ///////////////////////////////////////////
        float conv2_output_transpose[num_primary_caps * dim_primary_caps];
        for (int i = 0; i < dim_primary_caps; i++) {
            for (int j = 0; j < num_primary_caps; j++) {
                //conv2_output_transpose[j * dim_primary_caps + i] = conv2_output[i * num_primary_caps + j];
                conv2_output_transpose[i * num_primary_caps + j] = conv2_output[i * num_primary_caps + j];
            }
        }
        

        squash__(conv2_output_transpose, conv2_output, dim_primary_caps, num_primary_caps);

        prediction_vectors(conv2_output, digits_W, u_hat);



        dynamic_routing(u_hat, digits_bias, result_v);



        max = 0;
        for (int i = 0; i < num_class; i++) {
            predict_sum = 0;
            for (int j = 0; j < dim_predic_vector; j++) {
                predict_sum += pow(result_v[i * dim_predic_vector + j], 2);
            }
            if (predict_sum > max) {
                max = predict_sum;
                predict = i;
            }
        }

        printf("(%d) pred : %d / ", k + 1, predict);
        if (predict == (int)*(LABEL + k))  correct++;
        else {
            wrong++;
            num_wrong = fopen("error.txt", "wb");
            fprintf(num_wrong, "error : %d, wrong : %d\n", k, wrong);
            fclose(num_wrong);
        }
        printf("target: %d / ", (int)*(LABEL + k));
        accuracy = (float)correct / (k + 1);
        printf("accuracy : %.2f\n\n", accuracy * 100);

    }
    fclose(fp);
    return 0;
}
